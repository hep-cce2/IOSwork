#!/usr/bin/env python3
"""
Sanity-check the data volumes written by the ROOT cluster dumper (TTree and RNTuple).

usage:
  python verify_root_dump.py input.root dump.json            # with ROOT: compares against the file
  python verify_root_dump.py --no-root --file-size N dump.json   # JSON-only (no ROOT needed)

JSON-only checks (tested on the EVNT TTree and AOD RNTuple dumps):
  TTree   : sum(clusters) == sum(branch_totals); per-cluster branch sums; cursor overlap
  RNTuple : sum(clusters) == sum(fields) == sum(pages); pages shared by several columns
            (same offset+end) -> physical bytes vs dumped bytes; offset-0 ("virtual") pages;
            field_totals vs clusters; overlap between DISTINCT page ranges; coverage of file size
With ROOT (UNTESTED, no ROOT was available when written) it also compares with
  TTree   : sum of GetBasketBytes() over ALL (sub-)branches, tree.GetZipBytes(), true GetBasketSeek ranges
  RNTuple : RNTupleInspector.GetCompressedSize() (a per-column sum, must equal the dump)
"""
import argparse, collections, json, sys

OK, WARN, BAD = "OK", "WARN", "MISMATCH"


def fmt(n):
    return f"{n:,}"


# ----------------------------------------------------------------------------- TTree
def tree_json_checks(name, d):
    a = sum(c["total_bytes"] for c in d["clusters"])
    notes, verdict = [], OK
    if "branch_totals" in d:
        bt = sum(v["bytes"] for v in d["branch_totals"].values())
        if bt != a:
            verdict = BAD; notes.append(f"branch_totals {fmt(bt)} != clusters {fmt(a)}")
    for c in d["clusters"]:
        if "branches" in c and sum(b["bytes"] for b in c["branches"]) != c["total_bytes"]:
            verdict = BAD; notes.append(f"cluster {c['cluster_index']}: branch sum != total_bytes"); break
    return a, verdict, notes


def tree_root_checks(f, name, d):
    import ROOT

    def walk(branches):
        for b in branches:
            yield b
            yield from walk(b.GetListOfBranches())

    def baskets(b):
        n = b.GetWriteBasket()                    # baskets actually written (GetMaxBaskets may be larger)
        return [(int(b.GetBasketSeek(i)), int(b.GetBasketBytes()[i])) for i in range(n)]

    t = f.Get(name)
    top = sum(sz for b in t.GetListOfBranches() for _, sz in baskets(b))
    allb = [x for b in walk(t.GetListOfBranches()) for x in baskets(b)]
    return top, sum(sz for _, sz in allb), int(t.GetZipBytes()), [(o, o + sz, name) for o, sz in allb if sz > 0]


# --------------------------------------------------------------------------- RNTuple
def rntuple_json_checks(name, d):
    """Returns dict of numbers + list of (start,end,name) for non-virtual pages."""
    clusters = sum(c["total_bytes"] for c in d["clusters"])
    fields = pages_sum = 0
    ranges = collections.Counter()
    zero_pages = zero_bytes = 0
    per_page_mismatch = 0
    for c in d["clusters"]:
        for fl in c.get("fields", []):
            fields += fl["compressed_bytes"]
            ps = sum(p["compressed_bytes"] for p in fl["pages"])
            pages_sum += ps
            per_page_mismatch += ps != fl["compressed_bytes"]
            for p in fl["pages"]:
                if p["offset"] == 0:                           # not a real file position
                    zero_pages += 1; zero_bytes += p["compressed_bytes"]; continue
                ranges[(p["offset"], p["end"])] += 1
    distinct = sum(e - s for s, e in ranges)
    shared_extra = sum((e - s) * (n - 1) for (s, e), n in ranges.items() if n > 1)
    shared_ranges = sum(1 for n in ranges.values() if n > 1)
    ft = sum(v["compressed_bytes"] for v in d.get("field_totals", {}).values())
    return dict(clusters=clusters, fields=fields, pages=pages_sum, distinct=distinct,
                shared_extra=shared_extra, shared_ranges=shared_ranges, n_ranges=len(ranges),
                zero_pages=zero_pages, zero_bytes=zero_bytes, field_totals=ft,
                top=d.get("compressed_bytes"), page_mismatch=per_page_mismatch), \
        [(s, e, name) for (s, e) in ranges]


_POS_TYPE = None


def locator_position(loc):
    """File offset of an RNTupleLocator. In ROOT >= 6.34 the position is a
    std::variant<std::uint64_t, RNTupleLocatorObject64>; std::uint64_t is 'unsigned long' on
    Linux x86-64, so 'unsigned long long' does NOT instantiate (cppyy prints a long compile error).
    The type that works is cached so a failed instantiation is tried at most once."""
    global _POS_TYPE
    import ROOT
    if _POS_TYPE is not None:
        return int(loc.GetPosition[_POS_TYPE]())
    for typ in (ROOT.std.uint64_t, "std::uint64_t", "unsigned long"):
        try:
            v = int(loc.GetPosition[typ]())
            _POS_TYPE = typ
            return v
        except Exception:
            continue
    return 0                                              # non-file locator (e.g. object64/DAOS)


def rntuple_root_checks(name, input_file):
    import ROOT
    insp = ROOT.ROOT.Experimental.RNTupleInspector.Create(name, input_file)
    reader = ROOT.RNTupleReader.Open(name, input_file)
    desc = reader.GetDescriptor()
    seen, total = set(), 0
    for cid in desc.GetClusterIterable():
        cd = desc.GetClusterDescriptor(cid.GetId())
        for colrange in cd.GetColumnRangeIterable():
            for pi in cd.GetPageRange(colrange.GetPhysicalColumnId()).GetPageInfos():
                loc = pi.GetLocator()
                key = (locator_position(loc), int(loc.GetNBytesOnStorage()))
                if key[0] and key not in seen:
                    seen.add(key); total += key[1]
    return int(insp.GetCompressedSize()), total


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="[input.root] dump.json")
    ap.add_argument("--no-root", action="store_true")
    ap.add_argument("--file-size", type=int, default=None)
    args = ap.parse_args()
    args.dump = args.paths[-1]
    args.root_file = args.paths[0] if len(args.paths) > 1 else None
    if args.root_file is None and not args.no_root:
        ap.error("give input.root and dump.json, or use --no-root")
    dump = json.load(open(args.dump))
    cols = dump.get("collections", {dump.get("tree_name", "?"): dump})
    input_file = args.root_file if not args.no_root else None
    f, file_size = None, args.file_size
    if input_file:
        import ROOT
        f = ROOT.TFile.Open(input_file)
        file_size = file_size or int(f.GetSize())

    problems = 0
    tree_spans, rn_spans, cursor_starts = [], [], []

    ttrees = {n: d for n, d in cols.items() if d.get("type") == "TTree"}
    if ttrees:
        print("== TTree collections ==")
        print(f"{'collection':26s} {'dumped A':>15s} {'top B':>15s} {'all C':>15s} {'ZipBytes D':>15s}  verdict")
        for n, d in ttrees.items():
            a, verdict, notes = tree_json_checks(n, d)
            b = c = z = None
            if f:
                b, c, z, sp = tree_root_checks(f, n, d)
                tree_spans += sp
                if not (a == c == z):
                    verdict = "MISSING SUB-BRANCHES" if a == b < c else BAD
                    if a < c:
                        notes.append(f"{fmt(c - a)} bytes of baskets not in the dump "
                                     f"(e.g. baskets starting inside the last cluster, or sub-branches)")
            cursor_starts.append((n, d["clusters"][0].get("start_byte", 0)))
            problems += verdict != OK
            row = lambda v: f"{fmt(v):>15s}" if v is not None else f"{'-':>15s}"
            print(f"{n:26s} {row(a)} {row(b)} {row(c)} {row(z)}  {verdict}")
            for x in notes:
                print(f"    - {x}")
        if len(ttrees) > 1 and all(s == 0 for _, s in cursor_starts):
            print(f"  WARN: all {len(ttrees)} TTree collections start their start_byte cursor at 0, so the dumped "
                  f"byte ranges overlap between collections (they are not file offsets).")
        if tree_spans:
            tree_spans.sort()
            over = sum(max(0, min(e1, e2) - s2) for (s1, e1, _), (s2, e2, _) in zip(tree_spans, tree_spans[1:]) if s2 < e1)
            tot = sum(e - s for s, e, _ in tree_spans)
            print(f"  true basket ranges: {len(tree_spans):,} baskets, {fmt(tot)} bytes, overlap {fmt(over)} (expected 0)")
            if file_size:
                print(f"  file size {fmt(file_size)}: {fmt(file_size - tot)} bytes "
                      f"({(file_size - tot) / file_size:.2%}) are not basket data (headers, keys, ...)")

    rnts = {n: d for n, d in cols.items() if "RNTuple" in str(d.get("type"))}
    if rnts:
        print("\n== RNTuple collections ==")
        print(f"{'collection':18s} {'dumped A':>15s} {'physical B':>15s} {'shared extra':>13s} {'virt(off=0)':>12s} "
              f"{'field_tot':>14s} {'inspector D':>15s}  verdict")
        for n, d in rnts.items():
            r, sp = rntuple_json_checks(n, d)
            rn_spans += sp
            notes, verdict = [], OK
            if not (r["clusters"] == r["fields"] == r["pages"]):
                verdict = BAD; notes.append("clusters / fields / pages sums disagree")
            if r["page_mismatch"]:
                verdict = BAD; notes.append(f"{r['page_mismatch']} fields whose pages do not add up to compressed_bytes")
            if r["field_totals"] != r["clusters"]:
                verdict = WARN if verdict == OK else verdict
                notes.append(f"field_totals sums to {fmt(r['field_totals'])} != {fmt(r['clusters'])} "
                             f"(entries overwritten because it is keyed by field name)")
            if "physical_compressed_bytes" in d:          # patched dumper: it reports its own physical numbers
                for key, mine in (("physical_compressed_bytes", r["distinct"]),
                                  ("shared_pages_extra_bytes", r["shared_extra"]),
                                  ("virtual_bytes", r["zero_bytes"]), ("virtual_pages", r["zero_pages"])):
                    if d[key] != mine:
                        verdict = BAD; notes.append(f"dump {key} = {fmt(d[key])} but pages give {fmt(mine)}")
            elif r["shared_extra"]:
                verdict = WARN if verdict == OK else verdict
                notes.append("old dump format: no physical-volume keys; shared pages are counted once per column")
            if r["shared_extra"]:
                notes.append(f"info: {fmt(r['shared_extra'])} bytes ({r['shared_extra'] / r['clusters']:.2%}) of pages are shared by "
                             f"several columns ({r['shared_ranges']:,} of {r['n_ranges']:,} distinct page ranges); "
                             f"dumped = logical per-column bytes, physical = {fmt(r['distinct'])}")
            if r["zero_pages"]:
                notes.append(f"info: {r['zero_pages']} pages with offset 0 ({fmt(r['zero_bytes'])} bytes), "
                             f"not real file positions; excluded from physical")
            insp = None
            if input_file:
                insp, walk = rntuple_root_checks(n, input_file)
                # The inspector sums the compressed size of every column, so it also counts a page shared
                # by several columns once per column: it must equal the dump (A), not the physical size (B).
                if insp != r["clusters"]:
                    verdict = BAD; notes.append(f"dump {fmt(r['clusters'])} != ROOT inspector {fmt(insp)}")
                else:
                    notes.append("info: dump equals ROOT's per-column sum (RNTupleInspector)")
            problems += verdict != OK
            print(f"{n:18s} {fmt(r['clusters']):>15s} {fmt(r['distinct']):>15s} {fmt(r['shared_extra']):>13s} "
                  f"{fmt(r['zero_bytes']):>12s} {fmt(r['field_totals']):>14s} {(fmt(insp) if insp is not None else '-'):>15s}  {verdict}")
            for x in notes:
                print(f"    - {x}")
        rn_spans = sorted(set(rn_spans))
        over = 0; last = -1
        for s, e, _ in rn_spans:
            if s < last:
                over += min(last, e) - s
            last = max(last, e)
        phys = sum(e - s for s, e, _ in set((s, e, "") for s, e, _ in rn_spans))
        print(f"  distinct page ranges over all RNTuple collections: {fmt(phys)} bytes, "
              f"overlap between distinct ranges {fmt(over)}")
        if file_size:
            gap = file_size - phys
            print(f"  file size {fmt(file_size)}: {fmt(gap)} bytes ({gap / file_size:.2%}) are not page data "
                  f"(header/footer/anchors){'  <-- NEGATIVE: pages over-counted' if gap < 0 else ''}")
    if not input_file:
        print("\nJSON-only mode: internal consistency only. Basket/page volumes were NOT compared with the ROOT file "
              "(e.g. baskets missing from a TTree dump cannot be seen here); run with input.root for that.")
    print(f"\n{problems} collection(s) with issues")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())