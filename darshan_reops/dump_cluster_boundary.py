import ROOT
import sys
import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Optimized branch processor (TTree)
# -----------------------------
class BranchBasketBytes:
    """
    Basket sizes / first entries / FILE positions of one branch.
    Uses GetWriteBasket() (baskets really written) instead of GetMaxBaskets(): the unused tail of
    the ROOT arrays is zero-filled, which breaks np.searchsorted and (for small trees) was counted
    as phantom baskets.
    """
    def __init__(self, branch: ROOT.TBranch):
        n = int(branch.GetWriteBasket())
        self.branchName = str(branch.GetFullName())     # TString in recent ROOT: must be a plain str for json
        self.n = n
        if n:
            self.firstEntry = np.frombuffer(branch.GetBasketEntry(), dtype=np.int64, count=n).copy()
            self.sizes = np.frombuffer(branch.GetBasketBytes(), dtype=np.int32, count=n).astype(np.int64)
            self.seeks = np.array([int(branch.GetBasketSeek(i)) for i in range(n)], dtype=np.int64)
        else:
            self.firstEntry = self.sizes = self.seeks = np.zeros(0, dtype=np.int64)
        self.cluster_idx = None

    def assign(self, cluster_begins):
        """Every basket belongs to the cluster containing its first entry; nothing is dropped.
        Returns (bytes_per_cluster, baskets_per_cluster)."""
        ncl = len(cluster_begins)
        if self.n == 0:
            self.cluster_idx = np.zeros(0, dtype=np.int64)
            return np.zeros(ncl, dtype=np.int64), np.zeros(ncl, dtype=np.int64)
        idx = np.searchsorted(cluster_begins, self.firstEntry, side='right') - 1
        self.cluster_idx = np.clip(idx, 0, ncl - 1)
        return (np.bincount(self.cluster_idx, weights=self.sizes, minlength=ncl).astype(np.int64),
                np.bincount(self.cluster_idx, minlength=ncl).astype(np.int64))

    def file_ranges(self, k):
        """True [start, end) file ranges of the baskets of cluster k, adjacent ranges merged."""
        lo = np.searchsorted(self.cluster_idx, k, side='left')
        hi = np.searchsorted(self.cluster_idx, k, side='right')
        order = np.argsort(self.seeks[lo:hi], kind='stable')
        out = []
        for st, sz in zip(self.seeks[lo:hi][order].tolist(), self.sizes[lo:hi][order].tolist()):
            if sz <= 0:
                continue
            if out and out[-1][1] == st:
                out[-1][1] = st + sz
            else:
                out.append([st, st + sz])
        return out


def _all_branches(branches):
    """Top-level branches and, recursively, all sub-branches (their baskets hold data too)."""
    for br in branches:
        yield br
        yield from _all_branches(br.GetListOfBranches())

# -----------------------------
# Optimized field processor (RNTuple)
# -----------------------------
class RNTupleFieldMap:
    """
    Owns all per-field metadata extracted from an RNTuple descriptor and
    provides the same bytesInNextCluster interface as BranchBasketBytes so
    that the two storage backends can be driven by identical calling code.

    Responsibilities (previously split across three separate pieces):
      • _build_field_name_map  → built in __init__ via _walk_fields()
      • _get_locator_offset    → encapsulated in _locator_offset()
      • FieldPageBytes logic   → bytesInNextCluster() / page numpy arrays
    """

    # ROOT uses std::uint32_t max as the invalid-field sentinel
    _INVALID_FID = 0xFFFFFFFF

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, desc, field_id: int, field_name: str,
                 pages_data: list[tuple]):
        """
        Parameters
        ----------
        desc        : RNTupleDescriptor  – needed only to resolve the field
                      name map; stored as a weak reference so we don't keep
                      the entire descriptor alive longer than necessary.
        field_id    : int
        field_name  : str  – fully-qualified dotted name built by build_name_map()
        pages_data  : list of (first_entry, num_entries, saved_bytes)
        """
        self.fieldId   = field_id
        self.fieldName = field_name

        # numpy arrays for O(log n) cluster slicing (mirrors BranchBasketBytes)
        self.pageFirstEntry = np.array([p[0] for p in pages_data], dtype=np.int64)
        self.pageEntries    = np.array([p[1] for p in pages_data], dtype=np.int64)
        self.pageBytes      = np.array([p[2] for p in pages_data], dtype=np.int64)

        self.maxPages  = len(pages_data)
        self.iPage     = 0
        self.isAligned = True

    # ------------------------------------------------------------------ #
    #  Class-level factory: build the full name map from a descriptor     #
    # ------------------------------------------------------------------ #

    @classmethod
    def build_name_map(cls, desc) -> dict[int, str]:
        """
        Recursively walk every field in *desc* and return
            { field_id (int) : 'parent.child.grandchild' (str) }

        Replaces the old _build_field_name_map() free function.
        Does not depend on kInvalidFieldId being importable from cppyy;
        uses the literal sentinel 0xFFFFFFFF instead.
        """
        name_map: dict[int, str] = {}

        def _walk(fid: int, prefix: str) -> None:
            if fid == cls._INVALID_FID:
                return
            fd = desc.GetFieldDescriptor(fid)
            qn = f"{prefix}.{fd.GetFieldName()}" if prefix else fd.GetFieldName()
            name_map[int(fid)] = qn
            for child_id in fd.GetLinkIds():
                _walk(int(child_id), qn)

        root_fd = desc.GetFieldZero()
        for top_id in root_fd.GetLinkIds():
            _walk(int(top_id), "")

        return name_map

    # ------------------------------------------------------------------ #
    #  Static helper: resolve a template GetPosition<T>() call            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def locator_offset(loc) -> int:
        """
        Extract the byte offset from an RNTupleLocator.

        GetPosition<T>() is a C++ template; cppyy requires an explicit type
        argument.  Try uint64_t first (correct for files > 2 GB), then two
        fallback spellings that cover older ROOT / cppyy builds.

        Replaces the old _get_locator_offset() free function.
        """
        # In ROOT >= 6.34 the position is std::variant<std::uint64_t, RNTupleLocatorObject64>;
        # std::uint64_t is 'unsigned long' on Linux, so 'unsigned long long' does not instantiate.
        for typ in (ROOT.std.uint64_t, 'unsigned long'):
            try:
                return int(loc.GetPosition[typ]())
            except (TypeError, AttributeError):
                continue
        raise RuntimeError(
            f"Cannot instantiate GetPosition() on locator of type {type(loc)}"
        )

    # ------------------------------------------------------------------ #
    #  Instance method: same interface as BranchBasketBytes               #
    # ------------------------------------------------------------------ #

    def bytesInNextCluster(self, clusterBegin: int,
                           clusterEnd: int) -> tuple[int, int, bool]:
        """
        Return (bytes, page_count, is_aligned) for the pages that fall
        inside [clusterBegin, clusterEnd).

        Mirrors BranchBasketBytes.bytesInNextCluster() exactly so both
        can be iterated with the same loop in clusterPrintRNTuple.
        """
        if self.iPage >= self.maxPages:
            return (0, 0, True)

        if self.pageFirstEntry[self.iPage] > clusterBegin:
            self.isAligned = False
            return (0, 0, True)

        rel_end = np.searchsorted(
            self.pageFirstEntry[self.iPage:self.maxPages],
            clusterEnd, side='left'
        )
        end_idx = self.iPage + rel_end

        bytes_sum   = int(np.sum(self.pageBytes[self.iPage:end_idx]))
        pages_count = end_idx - self.iPage

        if end_idx < self.maxPages and self.pageFirstEntry[end_idx] < clusterEnd:
            self.isAligned = False

        self.iPage = end_idx
        return (bytes_sum, pages_count, self.isAligned)

def clusterPrintTTree(tree, input_file, max_workers=None, per_branch=False):
    """Processes TTree to get cluster boundaries with formatted JSON output.

    start_byte/end_byte are running per-tree CURSORS (back-compatible, they overlap between
    collections). With per_branch=True every branch entry also carries "file_ranges": the true
    [start, end) file positions of its baskets in that cluster."""
    branch_processors = [BranchBasketBytes(b) for b in _all_branches(tree.GetListOfBranches())]
    entries = int(tree.GetEntries())

    iterator = tree.GetClusterIterator(0)
    begins, ends = [], []
    cb = iterator()
    while cb < entries:
        nxt = iterator()
        ce = entries if nxt <= cb else nxt
        begins.append(int(cb)); ends.append(int(ce))
        cb = ce
    cluster_begins = np.array(begins, dtype=np.int64)
    ncl = len(begins)

    per_bp = {bp.branchName: bp.assign(cluster_begins) for bp in branch_processors}

    results = []
    current_byte_offset = 0
    branch_totals_data = {bp.branchName: {"entries": entries, "bytes": 0, "baskets": 0,
                                          "start_byte": 0, "last_end_byte": 0,
                                          "file_start": None, "file_end": 0}
                          for bp in branch_processors}

    for k in range(ncl):
        cluster_bytes = 0
        cluster_baskets = 0
        branch_info_list = []
        cluster_branch_offset = current_byte_offset

        for bp in branch_processors:
            b_bytes = int(per_bp[bp.branchName][0][k])
            b_baskets = int(per_bp[bp.branchName][1][k])
            b_start = cluster_branch_offset
            b_end = b_start + b_bytes - 1 if b_bytes > 0 else b_start
            tot = branch_totals_data[bp.branchName]

            if k == 0:
                tot["start_byte"] = b_start
            tot["bytes"] += b_bytes
            tot["baskets"] += b_baskets
            if b_bytes > 0:
                tot["last_end_byte"] = b_end

            if per_branch:
                rng = bp.file_ranges(k)
                branch_info_list.append({
                    "branch": bp.branchName, "start_byte": b_start, "end_byte": b_end,
                    "bytes": b_bytes, "baskets": b_baskets, "file_ranges": rng
                })
                cluster_branch_offset += b_bytes
                if rng:
                    tot["file_start"] = rng[0][0] if tot["file_start"] is None else min(tot["file_start"], rng[0][0])
                    tot["file_end"] = max(tot["file_end"], max(r[1] for r in rng))

            cluster_bytes += b_bytes
            cluster_baskets += b_baskets

        total_bytes = int(cluster_bytes)
        cluster_data = {
            "cluster_index": k,
            "begin": begins[k], "end": ends[k],
            "entries": ends[k] - begins[k],
            "start_byte": current_byte_offset,
            "end_byte": current_byte_offset + total_bytes - 1 if total_bytes > 0 else current_byte_offset,
            "total_bytes": total_bytes, "max_baskets": int(cluster_baskets)
        }
        if per_branch: cluster_data["branches"] = branch_info_list

        results.append(cluster_data)
        current_byte_offset += total_bytes

    output = {"file_name": input_file, "type": "TTree", "tree_name": tree.GetName(),
              "total_entries": entries, "clusters": results,
              "offsets": "start_byte/end_byte are per-tree cursors; branches[].file_ranges are file positions"}

    if per_branch:
        final_branch_totals = {}
        for bp in branch_processors:
            b_sum = branch_totals_data[bp.branchName]
            final_branch_totals[bp.branchName] = {
                "entries": b_sum["entries"], "bytes": b_sum["bytes"], "baskets": b_sum["baskets"],
                "start_byte": b_sum["start_byte"],
                "end_byte": b_sum["last_end_byte"] if b_sum["bytes"] > 0 else b_sum["start_byte"],
                "file_start": b_sum["file_start"], "file_end": b_sum["file_end"],
            }
        output["branch_totals"] = final_branch_totals
    return output


def clusterPrintRNTuple(ntuple, tree_name, input_file,
                        max_workers=None, per_branch=False,
                        verbose=False):
    """
    Walk every cluster in an RNTuple and return a dict whose shape mirrors
    clusterPrintTTree so downstream code can treat both uniformly.

    TTree parallel:
        cluster.begin          → cluster.begin
        cluster.end            → cluster.end  (first_entry + n_entries)
        cluster.entries        → cluster.entries
        cluster.start_byte     → cluster.start_byte   (running byte cursor)
        cluster.end_byte       → cluster.end_byte
        cluster.total_bytes    → cluster.total_bytes  (compressed on-disk)
        cluster.max_baskets    → cluster.max_pages
        cluster.branches[]     → cluster.fields[]     (when per_branch=True)
        output.branch_totals   → output.field_totals
    """
    print(f"Processing RNTuple: {tree_name} from {input_file}")

    desc = ntuple.GetDescriptor()

    # ── compression (file-level) ───────────────────────────────────────────
    inspector = ROOT.ROOT.Experimental.RNTupleInspector.Create(
        tree_name, input_file
    )
    compression_string = str(inspector.GetCompressionSettingsAsString())

    if verbose:
        print(f"  Compression: {compression_string}")
        ntuple.PrintInfo(ROOT.ROOT.Experimental.ENTupleInfo.kStorageDetails)

    # ── field-id → qualified dotted name ──────────────────────────────────
    field_names = RNTupleFieldMap.build_name_map(desc)

    # ── per-field running totals (mirrors branch_totals in TTree output) ──
    # keyed by (col_id) so each physical column gets its own entry,
    # matching the one-row-per-branch structure of branch_totals
    field_totals_data  = {}   # col_id → accumulator dict
    field_order        = []   # insertion order for final output

    # ── running byte cursor across clusters (mirrors current_byte_offset) ─
    current_byte_offset = 0

    # ── cluster walk ──────────────────────────────────────────────────────
    clusters   = []
    page_owners = {}        # (offset, nbytes) -> [(col_id, field_name), ...]  (pages shared by columns)
    page_refs   = []        # (key, col_id, page_dict) to annotate after the walk
    physical_total = 0      # bytes of distinct, real (non-virtual) pages
    virtual_pages = virtual_bytes = 0
    nclusters  = desc.GetNClusters()
    cluster_id = desc.FindClusterId(0, 0)

    for n in range(nclusters):
        cd = desc.GetClusterDescriptor(cluster_id)

        first_entry = int(cd.GetFirstEntryIndex())
        n_entries   = int(cd.GetNEntries())
        begin       = first_entry
        end         = first_entry + n_entries

        if verbose:
            print(f"\nCluster {cluster_id}  ({n+1}/{nclusters})"
                  f"  rows [{begin}, {end})")

        cluster_total_bytes = 0
        cluster_total_pages = 0
        cluster_uncompressed = 0
        cluster_physical     = 0
        field_info_list     = []

        # byte cursor within this cluster (for per-field start/end bytes)
        cluster_field_offset = current_byte_offset

        # ── column / page walk ────────────────────────────────────────────
        for colrange in cd.GetColumnRangeIterable():
            col_id        = int(colrange.GetPhysicalColumnId())
            col_desc      = desc.GetColumnDescriptor(col_id)
            fid           = int(col_desc.GetFieldId())
            field_name    = field_names.get(fid, f"<unknown field {fid}>")
            column_type   = col_desc.GetType()
            bits_per_elem = int(col_desc.GetBitsOnStorage())

            # register in field_totals on first encounter
            if col_id not in field_totals_data:
                field_order.append(col_id)
                field_totals_data[col_id] = {
                    "field":                field_name,
                    "field_id":             fid,
                    "column_type":          column_type,
                    "entries":              int(desc.GetNEntries()),
                    "compressed_bytes":     0,
                    "uncompressed_bytes":   0,
                    "pages":                0,
                    "start_byte":           cluster_field_offset,
                    "last_end_byte":        cluster_field_offset,
                }

            try:
                page_range = cd.GetPageRange(col_id)
            except Exception as exc:
                if verbose:
                    print(f"    [skip col {col_id}] GetPageRange failed: {exc}")
                continue

            col_compressed   = 0
            col_uncompressed = 0
            col_pages        = 0
            pages_list       = []

            field_start_byte = cluster_field_offset

            for page_idx, pi in enumerate(page_range.GetPageInfos()):
                loc    = pi.GetLocator()
                length = int(loc.GetNBytesOnStorage())

                try:
                    offset = RNTupleFieldMap.locator_offset(loc)
                except RuntimeError as exc:
                    if verbose:
                        print(f"    [skip page {page_idx}] {exc}")
                    continue

                n_elems = int(pi.GetNElements())
                uncomp  = n_elems * bits_per_elem // 8
                is_comp = (length < uncomp) if uncomp > 0 else False

                col_compressed   += length
                col_uncompressed += uncomp
                col_pages        += 1

                # offset 0 cannot be a real page position (the file header lives there): virtual page
                virtual = (offset == 0)
                key = (offset, length)
                if virtual:
                    virtual_pages += 1
                    virtual_bytes += length
                else:
                    owners = page_owners.setdefault(key, [])
                    if not owners:                         # first column to reference this page
                        cluster_physical += length
                        physical_total   += length
                    owners.append((col_id, field_name))

                if per_branch:
                    page_dict = {
                        "page_index":         page_idx,
                        "offset":             offset,
                        "end":                offset + length,
                        "compressed_bytes":   length,
                        "uncompressed_bytes": uncomp,
                        "n_elements":         n_elems,
                        "is_compressed":      is_comp,
                        "virtual":            virtual,
                    }
                    pages_list.append(page_dict)
                    if not virtual:
                        page_refs.append((key, col_id, page_dict))

                if verbose:
                    comp_label = (f"compressed → {uncomp}B"
                                  if is_comp else "uncompressed")
                    print(f"    page[{page_idx}]  "
                          f"offset={offset:#012x}  len={length}B  "
                          f"nelems={n_elems}  ({comp_label})")

            field_end_byte = (cluster_field_offset + col_compressed - 1
                              if col_compressed > 0
                              else cluster_field_offset)

            # update field totals
            ft = field_totals_data[col_id]
            ft["compressed_bytes"]   += col_compressed
            ft["uncompressed_bytes"] += col_uncompressed
            ft["pages"]              += col_pages
            if col_compressed > 0:
                ft["last_end_byte"] = field_end_byte

            if per_branch:
                field_info_list.append({
                    "field":              field_name,
                    "column_id":          col_id,
                    "field_id":           fid,
                    "column_type":        column_type,
                    "start_byte":         field_start_byte,
                    "end_byte":           field_end_byte,
                    "compressed_bytes":   col_compressed,
                    "uncompressed_bytes": col_uncompressed,
                    "pages":              pages_list,
                })

            cluster_field_offset += col_compressed
            cluster_total_bytes  += col_compressed
            cluster_total_pages  += col_pages
            cluster_uncompressed += col_uncompressed

        cluster_end_byte = (current_byte_offset + cluster_total_bytes - 1
                            if cluster_total_bytes > 0
                            else current_byte_offset)

        cluster_dict = {
            "cluster_index":      int(cluster_id),
            "begin":              begin,
            "end":                end,
            "entries":            n_entries,
            "start_byte":         current_byte_offset,
            "end_byte":           cluster_end_byte,
            "total_bytes":        cluster_total_bytes,         # compressed, mirrors TTree
            "uncompressed_bytes": cluster_uncompressed,
            "physical_bytes":     cluster_physical,            # distinct pages only (shared pages once)
            "max_pages":          cluster_total_pages,         # mirrors max_baskets
            "compression":        compression_string,
        }
        if per_branch:
            cluster_dict["fields"] = field_info_list

        clusters.append(cluster_dict)
        current_byte_offset += cluster_total_bytes
        cluster_id = desc.FindNextClusterId(cluster_id)

    # ── mark pages shared by several columns (same offset+size) ───────────
    for key, col_id, page_dict in page_refs:
        owners = page_owners[key]
        page_dict["shared_count"] = len(owners)
        page_dict["is_owner"] = (owners[0][0] == col_id)       # count physical bytes once, at the owner
        if len(owners) > 1:
            page_dict["shared_with"] = [nm for cid, nm in owners if cid != col_id][:8]

    # ── field_totals: mirrors branch_totals ───────────────────────────────
    field_totals = {}
    for col_id in field_order:
        ft = field_totals_data[col_id]
        col_ratio = (ft["uncompressed_bytes"] / ft["compressed_bytes"]
                     if ft["compressed_bytes"] > 0 else 1.0)
        ft_key = ft["field"] if ft["field"] not in field_totals else f'{ft["field"]}#col{col_id}'
        field_totals[ft_key] = {
            "entries":            ft["entries"],
            "compressed_bytes":   ft["compressed_bytes"],    # mirrors "bytes"
            "uncompressed_bytes": ft["uncompressed_bytes"],
            "compression_ratio":  round(col_ratio, 4),
            "pages":              ft["pages"],               # mirrors "baskets"
            "start_byte":         ft["start_byte"],
            "end_byte":           (ft["last_end_byte"]
                                   if ft["compressed_bytes"] > 0
                                   else ft["start_byte"]),
        }

    grand_compressed   = sum(ft["compressed_bytes"]   for ft in field_totals_data.values())
    grand_uncompressed = sum(ft["uncompressed_bytes"]  for ft in field_totals_data.values())
    grand_ratio        = (grand_uncompressed / grand_compressed
                          if grand_compressed > 0 else 1.0)

    output = {
        # ── top-level keys present in TTree output ──
        "file_name":     input_file,
        "type":          "RNTuple",
        "tree_name":     tree_name,
        "total_entries": int(desc.GetNEntries()),
        "clusters":      clusters,
        "field_totals":  field_totals,           # mirrors "branch_totals"
        # ── RNTuple-only extras ──
        "compression":          compression_string,
        "compressed_bytes":     grand_compressed,        # per-column sum (matches RNTupleInspector)
        "physical_compressed_bytes": physical_total,     # distinct real pages: shared pages once
        "shared_pages_extra_bytes":  grand_compressed - physical_total - virtual_bytes,
        "virtual_pages":        virtual_pages,           # offset 0: not real file positions
        "virtual_bytes":        virtual_bytes,
        "uncompressed_bytes":   grand_uncompressed,
        "compression_ratio":    round(grand_ratio, 4),
        "n_clusters":           nclusters,
    }

    return output

def _resolve_collections(f, spec):
    """
    Turn the CLI spec into a list of (name, classname) pairs.
      "Events"            -> [("Events", <class>)]
      "Events,Runs,Lumis" -> three entries
      "all" or "*"        -> every TTree/RNTuple key in the file
    Duplicate keys (multiple cycles) are collapsed to one entry.
    """
    if spec in ("all", "*"):
        found, seen = [], set()
        for key in f.GetListOfKeys():
            cls, name = key.GetClassName(), key.GetName()
            if name in seen:
                continue
            if cls == "TTree" or "RNTuple" in cls:
                seen.add(name)
                found.append((name, cls))
        return found

    found = []
    for name in dict.fromkeys(n.strip() for n in spec.split(",") if n.strip()):
        key = f.FindKey(name)
        if not key:
            print(f"Warning: '{name}' not found in {f.GetName()}, skipping",
                  file=sys.stderr)
            continue
        found.append((name, key.GetClassName()))
    return found


def _process_collection(f, input_file, name, classname, max_workers, per_branch):
    if classname == "TTree":
        return clusterPrintTTree(f.Get(name), input_file,
                                 max_workers=max_workers, per_branch=per_branch)
    if "RNTuple" in classname:
        reader = ROOT.RNTupleReader.Open(name, input_file)
        return clusterPrintRNTuple(reader, name, input_file,
                                   max_workers=max_workers, per_branch=per_branch)
    raise ValueError(f"Unsupported class '{classname}' for '{name}'")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.root name[,name2,...]|all "
              f"[max_workers] [per_branch=0/1] [output.json]")
        sys.exit(1)

    input_file  = sys.argv[1]
    spec        = sys.argv[2]
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    per_branch  = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    output_file = sys.argv[5] if len(sys.argv) > 5 else None

    f = ROOT.TFile.Open(input_file)
    if not f or f.IsZombie():
        print(f"Cannot open {input_file}", file=sys.stderr)
        sys.exit(1)

    collections = _resolve_collections(f, spec)
    if not collections:
        print("No matching TTree/RNTuple found.", file=sys.stderr)
        sys.exit(1)

    results, errors = {}, {}
    for name, classname in collections:
        try:
            results[name] = _process_collection(
                f, input_file, name, classname, max_workers, per_branch)
        except Exception as e:
            errors[name] = str(e)
            print(f"Failed to process '{name}': {e}", file=sys.stderr)
    f.Close()

    if not results:
        sys.exit(1)

    # Single collection requested explicitly: keep the original output shape
    # so existing consumers keep working.
    if len(collections) == 1 and spec not in ("all", "*"):
        result = next(iter(results.values()))
    else:
        result = {"file_name": input_file, "collections": results}
        if errors:
            result["errors"] = errors

    if output_file:
        with open(output_file, "w") as out_f:
            json.dump(result, out_f, indent=2)
    else:
        print(json.dumps(result, indent=2))
        
if __name__ == "__main__":
    main()