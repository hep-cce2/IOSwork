import sys
import argparse
import bisect
import math
from typing import Any, Union
import json

import darshan

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

import pandas as pd
import numpy as np

# Default width for printed / plotted branch names (0 = never shorten).
# Set from --name_width in main().
NAME_WIDTH = 70

def shorten_name(name, width):
    """
    Shorten one name to at most `width` characters, keeping BOTH the start
    (collection / container) and the end (the field that tells siblings apart):
        EventData/InDetTrackParticlesAux:.defin...CovMatrixDiag._0
    width <= 0 or a name that already fits is returned unchanged.
    """
    name = str(name)
    if width <= 0 or len(name) <= width:
        return name
    if width <= 8:
        return name[:width]
    budget = width - 3                       # room for "..."
    head = int(round(budget * 0.45))
    tail = budget - head
    return name[:head] + "..." + name[len(name) - tail:]

def unique_short_names(names, width=None):
    """
    Shorten every name to about `width` characters such that no two DIFFERENT
    names print the same. Names that still collide get more characters (in
    steps) until they differ, up to their full length. Returns a list parallel
    to `names`.
    """
    width = NAME_WIDTH if width is None else width
    names = [str(n) for n in names]
    if width <= 0:
        return names
    cur = {n: width for n in set(names)}
    short = {n: shorten_name(n, w) for n, w in cur.items()}
    while True:
        groups = {}
        for n, s in short.items():
            groups.setdefault(s, []).append(n)
        clashing = [g for g in groups.values() if len(g) > 1]
        if not clashing:
            break
        grew = False
        for g in clashing:
            for n in g:
                if cur[n] < len(n):
                    cur[n] = min(len(n), cur[n] + 6)
                    short[n] = shorten_name(n, cur[n])
                    grew = True
        if not grew:
            break
    return [short[n] for n in names]


def _has_file_offsets(data):
    """True when a collection's byte ranges are real file positions: TTree dumps with
    branches[].file_ranges, RNTuple dumps with per-page offsets. Older dumps only carry
    per-tree running cursors, which overlap between collections."""
    for cluster in data.get("clusters", []):
        if data.get("type", "TTree") == "RNTuple":
            return any(fd.get("pages") for fd in cluster.get("fields", []))
        return any("file_ranges" in bd for bd in cluster.get("branches", []))
    return False

DEFAULT_COLLECTION = "CollectionTree"

def order_by_time(seg_df, sort=False):
    """
    The reread analysis treats row order as time order: a read is a reread when an
    EARLIER ROW already covered its bytes, and it is credited to the later row (thread).
    DXT segments are not guaranteed to be listed by start_time (for example one thread's
    block can precede another's), so check, warn, and optionally stable-sort by start_time.
    """
    if seg_df is None or not len(seg_df) or 'start_time' not in seg_df.columns:
        return seg_df
    t = seg_df['start_time'].values
    n_back = int((t[1:] < t[:-1]).sum())
    if n_back == 0:
        return seg_df
    if sort:
        print(f"Sorting {len(seg_df)} segments by start_time (stable): {n_back} rows were out of "
              f"time order in the log.")
        return seg_df.sort_values('start_time', kind='stable').reset_index(drop=True)
    print(f"Warning: {n_back} of {len(seg_df)} segments are listed out of start_time order. "
          f"Rereads are credited by row order, so 'earlier' may not mean earlier in time; "
          f"use --sort_by_time to sort by start_time first.")
    return seg_df

def format_bytes(y, pos):
    if y == 0: return "0B"
    if not np.isfinite(y): return str(y)
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    epsilon = 1e-9
    temp_y_div = y
    div_i = 0
    while abs(temp_y_div % 1024) < epsilon and temp_y_div >= 1024 and div_i < len(size_name) - 1:
        temp_y_div /= 1024.0
        div_i += 1
    
    if div_i > 0 and abs(temp_y_div - round(temp_y_div)) < epsilon:
        y_fmt = round(temp_y_div)
        i_fmt = div_i
    else:
        temp_y_general = y
        i_general = 0
        while temp_y_general >= 1024 and i_general < len(size_name) - 1:
            temp_y_general /= 1024.0
            i_general += 1
        y_fmt = temp_y_general
        i_fmt = i_general

    if abs(y_fmt - round(y_fmt)) < epsilon:
        return f"{round(y_fmt):,}{size_name[i_fmt]}"
    else:
        return f"{y_fmt:,.1f}{size_name[i_fmt]}"


def _rows_from_dump(data, collection=None, prefix_names=False, stats=None):
    """Flatten ONE collection dict into DataFrame rows.

    Offset_kind is 'file' for true file positions (TTree file_ranges, RNTuple pages) and
    'cursor' for the per-tree running cursors of older dumps. RNTuple pages at offset 0 are
    not real file positions (the file header lives there) and are skipped; they are counted
    in `stats`."""
    data_type = data.get("type", "TTree")
    coll_name = collection or data.get("tree_name", "")

    def name_of(n):
        return f"{coll_name}/{n}" if prefix_names and coll_name else n

    rows = []
    for cluster in data.get("clusters", []):
        common = {
            'Collection': coll_name,
            'Cluster':    cluster["cluster_index"],
            'Begin':      cluster["begin"],
            'End':        cluster["end"],
            'Entries':    cluster["entries"],
        }
        defaults = {
            'Column_id': -1, 'Field_id': -1, 'Column_type': '',
            'Uncompressed_bytes': 0, 'Page_index': -1, 'N_elements': 0,
            'Is_compressed': False, 'Share': 1, 'Offset_kind': 'cursor',
        }

        if data_type == "RNTuple" and "fields" in cluster:
            for fd in cluster["fields"]:
                base = {**common, **defaults,
                        'Branch':      name_of(fd["field"]),
                        'Column_id':   fd.get("column_id", -1),
                        'Field_id':    fd.get("field_id", -1),
                        'Column_type': fd.get("column_type", "")}
                pages = fd.get("pages", [])
                if pages:
                    for p in pages:
                        if p["offset"] == 0:
                            if stats is not None:
                                stats['virtual_pages'] = stats.get('virtual_pages', 0) + 1
                                stats['virtual_bytes'] = stats.get('virtual_bytes', 0) + p["compressed_bytes"]
                            continue
                        rows.append({**base,
                            'Start_byte':         p["offset"],
                            'End_byte':           p["end"] - 1,
                            'Bytes':              p["compressed_bytes"],
                            'Uncompressed_bytes': p.get("uncompressed_bytes", 0),
                            'Baskets':            1,
                            'Page_index':         p["page_index"],
                            'N_elements':         p.get("n_elements", 0),
                            'Is_compressed':      p.get("is_compressed", False),
                            'Offset_kind':        'file',
                            'Granularity':        'page'})
                elif fd.get("compressed_bytes", 0) == 0:
                    continue        # no bytes on disk: its cursor position is not a file position
                else:
                    rows.append({**base,
                        'Start_byte':         fd.get("start_byte", 0),
                        'End_byte':           fd.get("end_byte", 0),
                        'Bytes':              fd.get("compressed_bytes", 0),
                        'Uncompressed_bytes': fd.get("uncompressed_bytes", 0),
                        'Baskets':            fd.get("n_pages", 1),
                        'Granularity':        'field'})

        elif "branches" in cluster:
            for bd in cluster["branches"]:
                ranges = bd.get("file_ranges")
                if ranges is not None:
                    # one row per contiguous [start, end) range of baskets on disk
                    for r_start, r_end in ranges:
                        rows.append({**common, **defaults,
                            'Branch':      name_of(bd["branch"]),
                            'Start_byte':  r_start,
                            'End_byte':    r_end - 1,
                            'Bytes':       r_end - r_start,
                            'Baskets':     1,
                            'Offset_kind': 'file',
                            'Granularity': 'basket_range'})
                else:
                    rows.append({**common, **defaults,
                        'Branch':      name_of(bd["branch"]),
                        'Start_byte':  bd["start_byte"],
                        'End_byte':    bd["end_byte"],
                        'Bytes':       bd["bytes"],
                        'Baskets':     bd.get("baskets", 1),
                        'Granularity': 'basket'})

        else:  # cluster-level fallback
            # Not prefixed: downstream plotting keys on the "cluster_" name.
            rows.append({**common, **defaults,
                'Branch':             f'cluster_{cluster["begin"]}_{cluster["end"]}',
                'Start_byte':         cluster["start_byte"],
                'End_byte':           cluster["end_byte"],
                'Bytes':              cluster["total_bytes"],
                'Uncompressed_bytes': cluster.get("uncompressed_bytes", 0),
                'Baskets':            cluster.get("max_baskets",
                                                  cluster.get("max_pages", 1)),
                'Granularity':        'cluster'})
    return rows


def _select_collections(data, requested):
    """
    Return [(name, collection_dict), ...] for either JSON layout.
      old flat shape      -> one entry, named by its tree_name
      new multi shape     -> filtered by `requested` (comma list or 'all')
    """
    if "collections" not in data:
        return [(data.get("tree_name", ""), data)]

    available = data["collections"]
    for name, msg in data.get("errors", {}).items():
        print(f"Warning: collection '{name}' failed during dump: {msg}")

    if requested is None:
        if len(available) == 1:
            return list(available.items())
        if DEFAULT_COLLECTION in available:
            exact = all(_has_file_offsets(d) for d in available.values())
            print(f"Note: tree/branch file holds {len(available)} collections; using the default "
                  f"collection '{DEFAULT_COLLECTION}'. Choose others with --collection "
                  f"NAME[,NAME2,...] or --collection all."
                  + (" All collections carry true file offsets, so --collection all maps the whole "
                     "file without double counting." if exact else ""))
            return [(DEFAULT_COLLECTION, available[DEFAULT_COLLECTION])]
        if all(_has_file_offsets(d) for d in available.values()):
            print(f"Note: tree/branch file holds {len(available)} collections without '{DEFAULT_COLLECTION}'; "
                  f"all carry true file offsets, so all of them are mapped (same as --collection all).")
            return list(available.items())
        print("Error: tree/branch file contains multiple collections and no "
              f"'{DEFAULT_COLLECTION}'; choose with --collection NAME[,NAME2,...] or --collection all.")
        print(f"Available: {list(available)}")
        sys.exit(1)

    if requested in ("all", "*"):
        return list(available.items())

    names = [n.strip() for n in requested.split(",") if n.strip()]
    missing = [n for n in names if n not in available]
    if missing:
        print(f"Error: collection(s) {missing} not in file. "
              f"Available: {list(available)}")
        sys.exit(1)
    return [(n, available[n]) for n in names]


def load_tree_branch_data(tree_branch_file, collection=None):
    """Load cluster/branch data from JSON.
    Supports TTree and RNTuple, in both the original single-collection
    output and the multi-collection ('collections') output."""
    try:
        with open(tree_branch_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading tree/branch data: {e}")
        return pd.DataFrame()

    selected = _select_collections(data, collection)   # may sys.exit on bad names
    multi = len(selected) > 1

    cursor_trees = [n for n, d in selected
                    if d.get("type", "TTree") == "TTree" and not _has_file_offsets(d)]
    if multi and cursor_trees:
        print("Warning: these TTree collections only carry per-tree cursors (no file_ranges), "
              f"which overlap between collections: {cursor_trees}. Mapping several of them is "
              "ambiguous (rereads are credited to every overlapping entry); select one collection "
              "or re-dump with the current dumper.")

    try:
        rows, dump_stats = [], {}
        for name, d in selected:
            rows.extend(_rows_from_dump(d, collection=name, prefix_names=multi, stats=dump_stats))

        df = pd.DataFrame(rows)
        if not df.empty:
            # RNTuple pages stored once but referenced by several columns (same offset and size):
            # a reread of such a page is split equally among the sharing fields (Share = number of
            # fields), so the credited bytes add up to the physical bytes read.
            is_page = (df['Granularity'] == 'page').values
            if is_page.any():
                grp = df[is_page].groupby(['Collection', 'Start_byte', 'End_byte'])['Branch'].transform('nunique')
                df.loc[is_page, 'Share'] = grp.values
                n_shared = int((df.loc[is_page, 'Share'] > 1).sum())
                if n_shared:
                    print(f"RNTuple: {n_shared:,} page records are shared by several fields; "
                          f"rereads on them are split equally among those fields.")
            if dump_stats.get('virtual_pages'):
                print(f"RNTuple: skipped {dump_stats['virtual_pages']:,} pages at offset 0 "
                      f"({dump_stats['virtual_bytes']:,} bytes): not real file positions.")
            kinds = {n: d.get("type", "TTree") for n, d in selected}
            print(f"Loaded DataFrame from {kinds}")
            print(f"Shape:       {df.shape}")
            print(f"Granularity: {df['Granularity'].value_counts().to_dict()}")
            print(f"Offsets:     {df['Offset_kind'].value_counts().to_dict()}")
            print(f"First few rows:\n{df.head()}")
        else:
            print("Warning: No data loaded from the file")
        return df
    except Exception as e:
        print(f"Error loading tree/branch data: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

def map_offsets_to_tree_branches(reop_data, tree_branch_df):
    """
    Map reread intervals to tree/branch/page entries using vectorized ops.
    Fixes 'First-Byte Winner' anomaly by accurately calculating the byte
    intersection of the Darshan physical read with the branch footprint.
    """
    offset_mapping = {}

    if tree_branch_df.empty:
        print("Warning: tree_branch_df is empty, no mapping can be performed.")
        return offset_mapping

    required_columns = ['Start_byte', 'Bytes', 'Cluster', 'Branch',
                        'Begin', 'End', 'Entries', 'Baskets']
    missing = [c for c in required_columns if c not in tree_branch_df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Available columns: {list(tree_branch_df.columns)}")
        return offset_mapping

    offsets = list(reop_data.keys())
    print(f"Mapping {len(offsets)} reread intervals to "
          f"{len(tree_branch_df)} tree/branch/page entries...")

    if 'End_byte' in tree_branch_df.columns:
        end_bytes = (tree_branch_df['End_byte']
                     .fillna(tree_branch_df['Start_byte']
                             + tree_branch_df['Bytes'] - 1) + 1)
    else:
        end_bytes = tree_branch_df['Start_byte'] + tree_branch_df['Bytes']

    start_bytes    = tree_branch_df['Start_byte'].values
    end_bytes      = end_bytes.values

    # pull optional RNTuple columns once, with safe fallbacks
    granularity_vals  = (tree_branch_df['Granularity'].values
                         if 'Granularity'  in tree_branch_df.columns
                         else np.full(len(tree_branch_df), 'unknown'))
    page_index_vals   = (tree_branch_df['Page_index'].values
                         if 'Page_index'   in tree_branch_df.columns
                         else np.full(len(tree_branch_df), -1))
    n_elements_vals   = (tree_branch_df['N_elements'].values
                         if 'N_elements'   in tree_branch_df.columns
                         else np.zeros(len(tree_branch_df)))
    column_id_vals    = (tree_branch_df['Column_id'].values
                         if 'Column_id'    in tree_branch_df.columns
                         else np.full(len(tree_branch_df), -1))
    field_id_vals     = (tree_branch_df['Field_id'].values
                         if 'Field_id'     in tree_branch_df.columns
                         else np.full(len(tree_branch_df), -1))
    column_type_vals  = (tree_branch_df['Column_type'].values
                         if 'Column_type'  in tree_branch_df.columns
                         else np.full(len(tree_branch_df), ''))
    uncomp_vals       = (tree_branch_df['Uncompressed_bytes'].values
                         if 'Uncompressed_bytes' in tree_branch_df.columns
                         else np.zeros(len(tree_branch_df)))
    is_comp_vals      = (tree_branch_df['Is_compressed'].values
                         if 'Is_compressed' in tree_branch_df.columns
                         else np.zeros(len(tree_branch_df), dtype=bool))
    share_vals        = (tree_branch_df['Share'].values.astype(float)
                         if 'Share' in tree_branch_df.columns
                         else np.ones(len(tree_branch_df)))

    for offset in offsets:
        info = reop_data[offset]
        parts = None
        if isinstance(info, dict) and 'bytes' in info:
            total_reop_bytes = info['bytes']
            ov_end = info['end']
            parts = info.get('parts')
        else:
            # Fallback if old structure
            total_reop_bytes = info
            ov_end = offset + 1

        ov_start = offset
        matches  = []

        # Vectorized check: does the reread interval intersect the branch?
        in_range = ((start_bytes < ov_end) & ((end_bytes) > ov_start))
        branch_indices = np.where(in_range)[0]

        # Exact attribution: every group of rereads (same start, same end) is spread over
        # ITS OWN [start, end) only. Spreading the summed bytes of all groups over the
        # longest one would hand short rereads' bytes to branches they never touched.
        attr_vec = None
        if parts and len(branch_indices):
            pe = np.fromiter(parts.keys(), dtype=np.float64, count=len(parts))
            pb = np.fromiter(parts.values(), dtype=np.float64, count=len(parts))
            inter = np.clip(np.minimum(pe[None, :], end_bytes[branch_indices][:, None].astype(np.float64))
                            - np.maximum(float(ov_start), start_bytes[branch_indices][:, None].astype(np.float64)),
                            0.0, None)
            attr_vec = (inter * (pb / (pe - ov_start))[None, :]).sum(axis=1)

        for k_match, branch_idx in enumerate(branch_indices):
            row        = tree_branch_df.iloc[branch_idx]
            b_start    = start_bytes[branch_idx]
            b_end      = end_bytes[branch_idx] # This is start + length

            # Calculate exact byte intersection
            intersect_start = max(ov_start, b_start)
            intersect_end   = min(ov_end, b_end)
            intersect_len   = max(0, intersect_end - intersect_start)

            # Attribute reoperation penalty proportionally to intersection size
            if attr_vec is not None:
                attributed_reop_bytes = float(attr_vec[k_match])
            else:
                ratio = intersect_len / (ov_end - ov_start) if (ov_end - ov_start) > 0 else 0
                attributed_reop_bytes = total_reop_bytes * ratio
            attributed_reop_bytes /= share_vals[branch_idx]      # shared RNTuple page: equal split

            match = {
                'collection':           str(row['Collection']) if 'Collection' in row.index else '',
                'cluster':              row['Cluster'],
                'branch':               row['Branch'],
                'begin':                row['Begin'],
                'end':                  row['End'],
                'entries':              row['Entries'],
                'start_byte':           int(b_start),
                'end_byte':             int(b_end - 1),
                'bytes':                row['Bytes'],
                'baskets':              row['Baskets'],
                'offset_within_branch': int(intersect_start - b_start),
                'total_reop_bytes':     total_reop_bytes,       # Total for this physical offset chunk
                'reoperation_bytes':    attributed_reop_bytes,  # True volumetric attribution for THIS branch
                'intersection_bytes':   int(intersect_len),     # Width of intersection
                # ── RNTuple extras ──────────────
                'granularity':          str(granularity_vals[branch_idx]),
                'page_index':           int(page_index_vals[branch_idx]),
                'n_elements':           int(n_elements_vals[branch_idx]),
                'column_id':            int(column_id_vals[branch_idx]),
                'field_id':             int(field_id_vals[branch_idx]),
                'column_type':          str(column_type_vals[branch_idx]),
                'uncompressed_bytes':   int(uncomp_vals[branch_idx]),
                'is_compressed':        bool(is_comp_vals[branch_idx]),
                'shared_count':         int(share_vals[branch_idx]),
            }
            matches.append(match)

        offset_mapping[offset] = matches

    return offset_mapping

def print_rntuple_suffix_explanation():
    """Prints a technical note explaining the _0 and _1 suffixes in RNTuple branch names."""
    explanation = """
------------------------------------------------------------
TECHNICAL NOTE: RNTuple Field Suffixes (e.g., _0, _1, etc.)
------------------------------------------------------------
In RNTuple mapping, numeric suffixes represent the physical 
flattening of C++ template containers (vectors, maps, pairs):

1. Suffix at the END (e.g., '.m_rawdata._0'):
   Represents the physical data column for primitive elements 
   inside a collection (e.g., std::vector<int>). High rereads 
   here indicate redundant loading of raw values.

2. Suffix in the MIDDLE (e.g., '._0.m_rowStrip'):
   Indicates a sub-field of a struct/class inside a collection.
   Rereads here suggest inefficient access to specific slices 
   of complex objects, often a target for I/O optimization.

3. Combined Suffixes (e.g., '._0._1'):
   Typically represents std::pair or std::map flattening.
   ._0 is the element (pair), and ._1 is the 'Value' (mapped_type).

Both cases contribute to 'Logical Reoperation Bytes' and should 
be tracked to identify specific physical column bottlenecks.
------------------------------------------------------------
"""
    print(explanation)

def _match_to_json(m, format_bytes, offset=None):
    """Serialize a full match dict to JSON-safe native types, keeping every field."""
    out = {
        'collection':                  str(m.get('collection', '')),
        'cluster':                     int(m['cluster']),
        'branch':                      str(m['branch']),
        'begin':                       int(m['begin']),
        'end':                         int(m['end']),
        'entries':                     int(m['entries']),
        'start_byte':                  int(m['start_byte']),
        'end_byte':                    int(m['end_byte']),
        'bytes':                       int(m['bytes']),
        'bytes_formatted':             format_bytes(m['bytes'], None),
        'baskets':                     int(m['baskets']),
        'offset_within_branch':        int(m['offset_within_branch']),
        'total_reop_bytes':            float(m['total_reop_bytes']),
        'total_reop_bytes_formatted':  format_bytes(m['total_reop_bytes'], None),
        'reoperation_bytes':           float(m['reoperation_bytes']),
        'reoperation_bytes_formatted': format_bytes(m['reoperation_bytes'], None),
        'intersection_bytes':          int(m['intersection_bytes']),
        'granularity':                 str(m['granularity']),
        'page_index':                  int(m['page_index']),
        'n_elements':                  int(m['n_elements']),
        'column_id':                   int(m['column_id']),
        'field_id':                    int(m['field_id']),
        'column_type':                 str(m['column_type']),
        'uncompressed_bytes':          int(m['uncompressed_bytes']),
        'uncompressed_bytes_formatted':format_bytes(m['uncompressed_bytes'], None),
        'is_compressed':               bool(m['is_compressed']),
        'shared_count':                int(m.get('shared_count', 1)),
    }
    if offset is not None:
        out['offset'] = int(offset)
        out['offset_formatted'] = format_bytes(offset, None)
    return out


def generate_mapping_report(offset_mapping, format_bytes,
                            output_prefix="output", op="operation",
                            tree_branch_df=None):
    """Generate detailed report and save to JSON. Print brief summary."""
    print("\n--- Reread Offset to Tree/Branch Mapping Report ---")

    total_mapped_offsets   = 0
    total_unmapped_offsets = 0
    branch_hit_count       = {}
    branch_reoperation_bytes = {}
    branch_cluster_indices = {}
    total_reoperation_bytes = 0
    detailed_mappings       = []

    # NEW: full-fidelity groupings
    matches_by_cluster = {}
    matches_by_branch  = {}

    for offset, matches in offset_mapping.items():
        offset_data = {
            "offset":                    int(offset),
            "offset_formatted":          format_bytes(offset, None),
            "reoperation_bytes":         0,
            "reoperation_bytes_formatted": "",
            "matches":                   []
        }

        if matches:
            total_mapped_offsets  += 1
            # We track total_reop_bytes exactly once per unique start offset chunk
            chunk_total_reop_bytes = matches[0].get('total_reop_bytes', 0)
            total_reoperation_bytes += chunk_total_reop_bytes

            offset_data["reoperation_bytes"]           = int(chunk_total_reop_bytes)
            offset_data["reoperation_bytes_formatted"] = format_bytes(chunk_total_reop_bytes, None)

            for match in matches:
                # Dump the ENTIRE match record (no truncation of fields)
                full_match = _match_to_json(match, format_bytes, offset=offset)
                offset_data["matches"].append(full_match)

                branch_name = full_match['branch']
                cluster_idx = full_match['cluster']

                branch_hit_count[branch_name] = branch_hit_count.get(branch_name, 0) + 1
                if branch_name not in branch_reoperation_bytes:
                    branch_reoperation_bytes[branch_name] = 0
                    branch_cluster_indices[branch_name]   = []

                # Fix: Accumulate exactly the intersection-attributed bytes, avoiding the first-byte anomaly
                branch_reoperation_bytes[branch_name] += full_match['reoperation_bytes']
                branch_cluster_indices[branch_name].append(cluster_idx)

                # NEW: group full match records by cluster
                matches_by_cluster.setdefault(str(cluster_idx), []).append(full_match)

                # NEW: group full match records by branch/field
                matches_by_branch.setdefault(branch_name, []).append(full_match)
        else:
            total_unmapped_offsets += 1
            offset_data["matches"] = None

        detailed_mappings.append(offset_data)

    is_per_branch = "per_branch" in output_prefix
    suffix        = "_per_branch" if is_per_branch else "_cluster"

    total_clusters = 0
    if tree_branch_df is not None and not tree_branch_df.empty:
        total_clusters = int(tree_branch_df['Cluster'].max() + 1)

    def analyze_temporal_distribution(cluster_indices, total_clusters):
        if total_clusters <= 1 or not cluster_indices:
            return "unknown"
        unique_clusters = sorted(set(cluster_indices))
        T1       = total_clusters * (1/3)
        T2       = total_clusters * (2/3)
        coverage = len(unique_clusters) / total_clusters
        min_c, max_c = min(unique_clusters), max(unique_clusters)
        if coverage >= 0.5:              return "throughout"
        if max_c < T1:                   return "early_localized"
        if min_c >= T2:                  return "late_localized"
        if min_c >= T1 and max_c < T2:  return "middle_localized"
        if min_c < T1  and max_c < T2:  return "early_to_middle"
        if min_c >= T1 and max_c >= T2: return "middle_to_late"
        if min_c < T1  and max_c >= T2: return "full_span"
        return "mixed_or_scattered"

    sorted_branches = sorted(
        [(b, branch_hit_count[b], branch_reoperation_bytes.get(b, 0))
         for b in branch_hit_count],
        key=lambda x: (x[2], x[1]), reverse=True
    )

    summary = {
        "total_offsets":        len(offset_mapping),
        "mapped_offsets":       total_mapped_offsets,
        "unmapped_offsets":     total_unmapped_offsets,
        "total_reoperation_bytes":           int(total_reoperation_bytes),
        "total_reoperation_bytes_formatted": format_bytes(total_reoperation_bytes, None),
        "total_clusters":       int(total_clusters),
        "branch_statistics": [
            {
                "branch":           str(branch),
                "hit_count":        int(count),
                "reoperation_bytes":            int(reop_bytes),
                "reoperation_bytes_formatted":  format_bytes(reop_bytes, None),
                "cluster_indices":  [int(x) for x in sorted(set(branch_cluster_indices[branch]))],
                "temporal_pattern": analyze_temporal_distribution(
                    branch_cluster_indices[branch], total_clusters)
            }
            for branch, count, reop_bytes in sorted_branches
        ]
    }

    # Sort each cluster's / branch's match list by offset for readability
    for cluster_key in matches_by_cluster:
        matches_by_cluster[cluster_key].sort(key=lambda m: m['offset'])
    for branch_key in matches_by_branch:
        matches_by_branch[branch_key].sort(key=lambda m: m['offset'])

    report = {
        "operation": op,
        "mode":      "per_branch" if is_per_branch else "cluster",
        "summary":   summary,
        "detailed_mappings":  detailed_mappings,     # per-offset view, full match fields
        "matches_by_cluster": matches_by_cluster,    # NEW: all full matches grouped by cluster
        "matches_by_branch":  matches_by_branch,     # NEW: all full matches grouped by branch/field
    }

    json_filename = f"{output_prefix}_mapping_report{suffix}.json"
    with open(json_filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Total Reread Intervals: {len(offset_mapping)} "
          f"(Mapped: {total_mapped_offsets}, Unmapped: {total_unmapped_offsets})")
    print(f"Total mapped Reoperation Bytes: {format_bytes(total_reoperation_bytes, None)}")
    print(f"Total Clusters: {total_clusters}")

    print_rntuple_suffix_explanation()

    print(f"{'Top 10 Branches/Fields by Reoperation Volume':<55}")
    top10 = sorted_branches[:10]
    top10_names = unique_short_names([b for b, _, _ in top10])
    name_col = max([len(s) for s in top10_names] + [1])
    for i, ((branch, count, reop_bytes), display) in enumerate(zip(top10, top10_names)):
        temporal = analyze_temporal_distribution(
            branch_cluster_indices[branch], total_clusters)
        print(f"  {i+1:2d}. {display:{name_col}s} : "
              f"{format_bytes(reop_bytes, None):>10s} ({count:3d} hits, {temporal})")

    print("\nTemporal Pattern Legend:")
    print("  throughout:          >=50% cluster coverage.")
    print("  early_localized:     First third of clusters (0–1/3).")
    print("  middle_localized:    Middle third of clusters (1/3–2/3).")
    print("  late_localized:      Last third of clusters (2/3–1).")
    print("  early_to_middle:     Spans first two thirds, low coverage.")
    print("  middle_to_late:      Spans last two thirds, low coverage.")
    print("  full_span:           Spans all three phases, low coverage.")
    print("  mixed_or_scattered:  Complex or non-major pattern.")
    print("  unknown:             Insufficient data (0 or 1 cluster).")
    print("--------------------------------------------------")
    
def compute_reoperation_bytes(df):
    """
    Optimized version using binary search and sorted interval tracking.
    Handles 100k+ rows efficiently.
    Returns both the dataframe and a mapping of offsets to their total reoperation bytes
    along with the specific end-boundaries of those overlaps.
    """
    starts = df['offset'].values.astype(int)
    lengths = df['length'].values.astype(int)
    start_times = df['start_time'].values
    ends = starts + lengths
    covered = []
    reoperation_bytes = []
    overlap_ctr = 0
    metadata_overlap_ctr = 0
    reop_offsets = set()
    reop_offsets_per_event = []
    offset_to_total_bytes = {}

    for i in range(len(starts)):
        start, end = starts[i], ends[i]
        overlap = 0
        ov_start = 0
        event_reop_offsets = set()
        idx = bisect.bisect_left(covered, (start,)) - 1
        idx = max(idx, 0)
        while idx < len(covered):
            s, e = covered[idx]
            if s >= end:
                break
            if e <= start:
                idx += 1
                continue
            ov_start = max(start, s)
            event_reop_offsets.add(ov_start)
            reop_offsets.add(ov_start)
            overlap_ctr += 1
            ov_end = min(end, e)
            overlap_amount = (ov_end - ov_start)
            overlap += overlap_amount

            # Store 'bytes' and 'end' safely to allow accurate range intersection mapping
            if ov_start not in offset_to_total_bytes:
                offset_to_total_bytes[ov_start] = {'bytes': 0, 'end': ov_end, 'parts': {}}
            offset_to_total_bytes[ov_start]['bytes'] += overlap_amount
            offset_to_total_bytes[ov_start]['end'] = max(offset_to_total_bytes[ov_start]['end'], ov_end)
            # Rereads that start at the same offset can end at different offsets (e.g. a
            # 300 B reread and a 4 MB reread of the same region). Keep the bytes per end
            # offset so the branch attribution can use each reread's own extent.
            parts = offset_to_total_bytes[ov_start]['parts']
            parts[ov_end] = parts.get(ov_end, 0) + overlap_amount

            if ov_start == 0:
                metadata_overlap_ctr += 1
            idx += 1
        reoperation_bytes.append(overlap)
        reop_offsets_per_event.append(event_reop_offsets)
        new_int = (start, end)
        insert_pos = bisect.bisect_left(covered, new_int)
        if insert_pos > 0 and covered[insert_pos-1][1] >= start:
            new_int = (covered[insert_pos-1][0], max(covered[insert_pos-1][1], end))
            covered.pop(insert_pos-1)
            insert_pos -= 1
        while insert_pos < len(covered) and covered[insert_pos][0] <= new_int[1]:
            new_int = (new_int[0], max(new_int[1], covered[insert_pos][1]))
            covered.pop(insert_pos)
        covered.insert(insert_pos, new_int)
    df['reoperation'] = reoperation_bytes
    df['reoperation_offsets'] = reop_offsets_per_event


    return df, reop_offsets, offset_to_total_bytes


def prepare_histogram_data(df, bins=30):
    all_offsets = []
    for index, row in df.iterrows():
        offsets = row['reoperation_offsets']
        for offset in offsets:
            all_offsets.append(offset)
    if not all_offsets:
        return None
    n, bins, patches = plt.hist(all_offsets, bins=bins, edgecolor='black', log=False)
    plt.close()
    temp_df = df.assign(original_event_idx=df.index)
    if not temp_df['reoperation_offsets'].apply(lambda x: isinstance(x, (list, set))).all():
        temp_df['reoperation_offsets'] = temp_df['reoperation_offsets'].apply(lambda x: list(x) if isinstance(x, (list, set)) else [] if pd.isna(x) else [x])
    flat_offsets_df = temp_df.explode('reoperation_offsets')
    if flat_offsets_df.empty:
        bin_overlapped_bytes_sums = [0.0] * (len(bins) - 1)
    else:
        flat_offsets_df['reoperation_offsets'] = pd.to_numeric(flat_offsets_df['reoperation_offsets'], errors='coerce')
        flat_offsets_df.dropna(subset=['reoperation_offsets'], inplace=True)
        flat_offsets_df['bin_idx'] = pd.cut(flat_offsets_df['reoperation_offsets'], bins=bins, labels=False, include_lowest=True, right=False)
        flat_offsets_df.dropna(subset=['bin_idx'], inplace=True)
        flat_offsets_df['bin_idx'] = flat_offsets_df['bin_idx'].astype(int)
        bin_overlapped_bytes_sums_series = flat_offsets_df.groupby(['bin_idx', 'original_event_idx'])['reoperation'].first().groupby(level=0).sum()
        bin_overlapped_bytes_sums = [0.0] * (len(bins) - 1)
        for i, val in bin_overlapped_bytes_sums_series.items():
            if 0 <= i < len(bin_overlapped_bytes_sums):
                bin_overlapped_bytes_sums[i] = val
    return {
        'all_offsets': all_offsets,
        'n': n,
        'bins': bins,
        'flat_offsets_df': flat_offsets_df,
        'bin_overlapped_bytes_sums': bin_overlapped_bytes_sums
    }


def report_top_bins_statistics(hist_data, df, format_bytes, top_bins=3, top_events=10, op="operation"):
    """
    Reports statistics for the top N histogram bins based on total overlapped bytes.
    """
    bin_overlapped_bytes_sums = hist_data['bin_overlapped_bytes_sums']
    bins = hist_data['bins']
    n = hist_data['n']
    flat_offsets_df = hist_data['flat_offsets_df']
    if bin_overlapped_bytes_sums:
        sorted_bins = sorted(enumerate(bin_overlapped_bytes_sums), key=lambda x: x[1], reverse=True)
        top_bins_info = sorted_bins[:top_bins]
        print(f"\n--- Statistics for Top {top_bins} Bins by Overlapped Bytes ({op}) ---")
        for rank, (bin_idx, overlapped_bytes_sum) in enumerate(top_bins_info):
            bin_start = bins[bin_idx]
            bin_end = bins[bin_idx + 1]
            num_reops_in_bin = int(n[bin_idx])
            print(f"\nRank {rank + 1} Bin:")
            print(f"  Bin Interval: [{format_bytes(bin_start, None)}, {format_bytes(bin_end, None)})")
            print(f"  Total Overlapped Bytes: {format_bytes(overlapped_bytes_sum, None)}")
            print(f"  Number of Re{op.capitalize()} Reoperations: {num_reops_in_bin}")
            events_in_this_bin_df = flat_offsets_df[flat_offsets_df['bin_idx'] == bin_idx].copy()
            events_details = df.loc[events_in_this_bin_df['original_event_idx'].unique()].copy()
            if 'reoperation' in events_details.columns:
                top_events_in_bin = events_details.sort_values(by='reoperation', ascending=False).head(top_events)
                if not top_events_in_bin.empty:
                    print(f"  Top {top_events} Events (Largest Overlapped Bytes) in this Bin:")
                    for _, event_row in top_events_in_bin.iterrows():
                        finishing_offset = event_row['offset'] + event_row['length']
                        print(f"    - Event Index: {event_row.name}, Overlapped: {format_bytes(event_row['reoperation'], None)}, Length: {format_bytes(event_row['length'], None)}, Start Time: {event_row['start_time']:.6f}s, End Time: {event_row['end_time']:.6f}s, Offsets: [{format_bytes(event_row['offset'], None)} - {format_bytes(finishing_offset, None)}] (Start Offset: {event_row['offset']})")
                else:
                    print("  No top events found for this bin.")
            else:
                print("  'reoperation' column not found in event details for sorting.")
        print("--------------------------------------------------")


def plot_offset_histogram(hist_data, title=None, output_prefix="output", op="operation"):
    """
    Generates and saves a histogram of overlapped offset values.
    """ 
    all_offsets = hist_data['all_offsets']
    n = hist_data['n']
    bins = hist_data['bins']
    bin_overlapped_bytes_sums = hist_data['bin_overlapped_bytes_sums']
    if not all_offsets:
        print(f"No overlapped offsets found to plot histogram for {op}.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    patches = ax.hist(all_offsets, bins=bins, edgecolor='black', log=False)[2]
    for i in range(len(patches)):
        patch = patches[i]
        count = n[i]
        sum_overlapped = bin_overlapped_bytes_sums[i]
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            label_text = format_bytes(sum_overlapped, None)
            ax.text(x, y, label_text, ha='center', va='bottom', fontsize=7, color='black')
    ax.set_xlim(left=0)
    ax.set_xlabel("Offset Values (Bytes)")
    ax.set_ylabel(f"Number of {op.capitalize()} Reoperations (Frequency)")
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.set_xticks(bin_centers)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    ax.tick_params(axis='x', rotation=40)
    if title is None:
        title = f"Histogram of Overlapped Offsets ({op.capitalize()})"
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(f"{output_prefix}_offset_histogram.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_reoperation_offsets_over_time(df, output_prefix="output", op="operation"):
    """
    Generates and saves a scatter plot visualizing reoperation offsets over time.
    """
    times = []
    offsets = []
    overlapped_bytes_list = []
    jitter = 0
    curr_ov_bytes = -1
    for idx, row in df.iterrows():
        overlap_bytes = row.get('reoperation', 1)
        for offset in row['reoperation_offsets']:
            times.append(row['start_time'])
            offsets.append(offset)
            overlapped_bytes_list.append(overlap_bytes)
    if not overlapped_bytes_list:
        print(f"No data to plot for {op}.")
        return

    if 0 in offsets:
        time_range = max(times) - min(times) if times else 1
        zero_offset_indices = [i for i, offset in enumerate(offsets) if offset == 0]
        for i in zero_offset_indices:
            times[i] += np.random.uniform(-0.02 * time_range, 0.02 * time_range)
    
    cmap = plt.cm.YlOrRd
    norm = LogNorm(vmin=1, vmax=max(overlapped_bytes_list))
    colorbar_label = 'Overlapped Bytes'
    colorbar_kws = {"label": colorbar_label}
    fig, ax = plt.subplots(figsize=(12, 5))
    fixed_point_size = 10
    sc = ax.scatter(times, offsets, alpha=1.0, s=fixed_point_size, c=overlapped_bytes_list, cmap=cmap, norm=norm)
    ax.set_xlabel('Start Time')
    ax.set_ylabel('Overlapping Offset')
    ax.set_title(f'Overlapping Offsets Over Time ({op.capitalize()})')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.margins(x=0)
    plt.tight_layout()
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, **colorbar_kws)
    fig.savefig(f"{output_prefix}_offset_over_time.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_branch_reoperation_bytes(offset_mapping, format_bytes, output_prefix="output", op="operation", top_n=-1, tree_branch_df=None, physical_reop_bytes=None, create_plot=True): 
    """
    Generates and saves a bar plot of reoperation bytes per branch/field.
    Uses corrected intersection attribution logic.
    """
    total_all_branches = 0
    if tree_branch_df is not None and not tree_branch_df.empty:
        total_all_branches = len(tree_branch_df['Branch'].unique())
    
    branch_reoperation_bytes = {}
    # Part of each branch's bytes that comes from RNTuple pages shared by several fields.
    # Those bytes are an equal-split estimate (the file cannot say which field caused the
    # read); everything else is exact.
    branch_shared_bytes = {}
    
    # Safely aggregate the newly calculated volumetric reoperation_bytes from matches
    for offset, matches in offset_mapping.items():
        if matches:
            for match in matches:
                branch_name = str(match['branch'])
                rb = match.get('reoperation_bytes', 0)
                branch_reoperation_bytes[branch_name] = branch_reoperation_bytes.get(branch_name, 0) + rb
                if int(match.get('shared_count', 1)) > 1:
                    branch_shared_bytes[branch_name] = branch_shared_bytes.get(branch_name, 0) + rb

    if not branch_reoperation_bytes:
        print(f"No branch reoperation data found to plot for {op}.")
        return
    
    # --top_n_plot: any value <= 0 means "all" (-1 is the documented default).
    top_n = top_n if top_n and top_n > 0 else -1

    # Use all active branches for JSON, independently of --top_n_plot. Totals are summed
    # from the exact (float) attributions and only rounded for display
    # Per-branch values stay floats (3 decimals): with shared pages the equal split gives
    # fractions, and rounding to whole bytes would hide them.
    non_empty_branches = [(name, round(float(value), 3)) for name, value in branch_reoperation_bytes.items()
                          if value > 1e-6]
    total_non_empty = len(non_empty_branches)
    if not non_empty_branches:
        print(f"No positive branch reoperation data found for {op}.")
        return
    stats_branches = sorted(non_empty_branches, key=lambda item: (-item[1], item[0]))
    # Distinguish physical rereads from their attribution to ROOT entries.
    # An offset with no matches, partially covered offsets, or overlapping
    # ROOT byte intervals can make these totals differ.
    mapped_offset_total = math.fsum(
        int(matches[0]['total_reop_bytes'])
        for matches in offset_mapping.values() if matches
    )
    attributed_total = math.fsum(float(v) for v in branch_reoperation_bytes.values() if v > 0)
    stats = {
        "operation": op,
        "total_branches": total_all_branches,
        "non_zero_branches": total_non_empty,
        "total_reop_bytes": int(round(attributed_total)),  # Legacy: ROOT-attributed sum, NOT physical total.
        "total_reop_bytes_meaning": "sum of proportional branch/page attributions (may differ from physical rereads)",
        "physical_reop_bytes": None if physical_reop_bytes is None else int(physical_reop_bytes),
        "mapped_offset_input_bytes": int(mapped_offset_total),
        "attributed_minus_mapped_offset_bytes": attributed_total - mapped_offset_total,
        "attributed_to_physical_ratio": (
            attributed_total / physical_reop_bytes if physical_reop_bytes else None
        ),
        "shared_page_reop_bytes": round(math.fsum(branch_shared_bytes.values()), 3),
        "shared_page_reop_bytes_meaning": "part of total_reop_bytes on RNTuple pages shared by several "
                                          "fields, split equally among them (an estimate)",
        "branches": [{"name": name, "bytes": value,
                      "shared_bytes": round(float(branch_shared_bytes.get(name, 0)), 3)}
                     for name, value in stats_branches],
    }
    if physical_reop_bytes is not None and not math.isclose(
        attributed_total, physical_reop_bytes, rel_tol=1e-8, abs_tol=1
    ):
        print("Warning: ROOT-attributed bytes do not equal physical reread bytes: "
              f"{attributed_total:.3f} vs {physical_reop_bytes}. "
              "Unmapped byte ranges, overlapping metadata ranges, and/or "
              "collapsed offset intervals may contribute.")
    with open(f"{output_prefix}_branch_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    if not create_plot:
        return

    is_cluster_mode = all(name.startswith("cluster_") for name, _ in non_empty_branches)
    if top_n > 0:
        branches_to_plot = stats_branches[:top_n]
    elif is_cluster_mode:
        branches_to_plot = sorted(non_empty_branches, key=lambda item: int(item[0].split('_')[1]))
    else:
        branches_to_plot = list(stats_branches)
    if not branches_to_plot:
        return

    if is_cluster_mode:
        branches_to_plot.sort(key=lambda x: int(x[0].split('_')[1]))
    
    if not branches_to_plot:
        return

    # 3. Prepare for Horizontal Bar Chart
    branches_to_plot.reverse()
    
    branch_names = [item[0] for item in branches_to_plot]
    reop_bytes = [item[1] for item in branches_to_plot]
    display_names = unique_short_names(branch_names,
                                       width=min(NAME_WIDTH, 50) if NAME_WIDTH > 0 else 0)
    
    # Figure width = room for the longest name + a plot area wide enough that the x tick
    # labels (about 8 characters each) do not overlap at the label font size.
    label_fs = 10
    char_in = label_fs * 0.62 / 72.0                     # approx. inches per character
    label_in = max((len(n) for n in display_names), default=10) * char_in + 0.6
    plot_in = 6.5
    n_xticks = max(3, int(plot_in / 1.15))
    fig_height = max(2.5, 1 + 0.3 * len(display_names))
    fig, ax = plt.subplots(figsize=(label_in + plot_in, fig_height))
    if top_n == -1 and is_cluster_mode:
        fig1, ax1 = plt.subplots(figsize=(14, 4))
    
    shared_bytes = [min(float(branch_shared_bytes.get(n, 0)), v) for n, v in zip(branch_names, reop_bytes)]
    has_shared = (not is_cluster_mode) and any(x > 0 for x in shared_bytes)
    if has_shared:
        # solid = exact part, hatched = part from shared pages (equal-split estimate)
        exact_bytes = [v - x for v, x in zip(reop_bytes, shared_bytes)]
        bars = ax.barh(range(len(display_names)), exact_bytes, color='steelblue', alpha=0.7,
                       edgecolor='black', linewidth=0.5)
        ax.barh(range(len(display_names)), shared_bytes, left=exact_bytes, color='white',
                edgecolor='#d62728', hatch='////', linewidth=0.8)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor='steelblue', alpha=0.7, edgecolor='black',
                                 label='exact (page used by this field only)'),
                           Patch(facecolor='white', edgecolor='#d62728', hatch='////',
                                 label='estimate (shared page, split equally)')],
                  loc='lower right', fontsize=8)
    else:
        bars = ax.barh(range(len(display_names)), reop_bytes, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=label_fs)
    ax.set_xlabel('Reoperation Volume', fontsize=12, labelpad=10)
    ax.set_ylabel('Cluster Index' if is_cluster_mode else 'Branch/Field Names', fontsize=12)
    
    title_suffix = f" - Top {len(branches_to_plot)}" if top_n != -1 else " (All Active)"
    ax.set_title(f'Reoperation Bytes by {"Cluster" if is_cluster_mode else "Branch"} ({op.capitalize()}){title_suffix}', 
                 fontsize=10, pad=10, fontweight='bold')
    
    # Nicer ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=n_xticks, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.grid(True, axis='x', which='major', linestyle='-', alpha=0.4)
    ax.grid(True, axis='x', which='minor', linestyle=':', alpha=0.2)
    
    max_width = max(reop_bytes) if reop_bytes else 0
    for bar, value, sh in zip(bars, reop_bytes, shared_bytes):
        # with the stacked bar, bar.get_width() is only the exact part: use the full value
        width = value
        if has_shared and sh > 0:
            ax.text(value + (max_width * 0.01), bar.get_y() + bar.get_height()/2,
                    f"{format_bytes(value, None)}  ({100.0 * sh / value:.0f}% shared)",
                    ha='left', va='center', fontsize=8, color='#d62728')
        elif width > max_width * 0.15:
            ax.text(width * 0.98, bar.get_y() + bar.get_height()/2, 
                   format_bytes(value, None), 
                   ha='right', va='center', fontweight='bold', fontsize=9, color='white')
        elif width > 0:
            ax.text(width + (max_width * 0.01), bar.get_y() + bar.get_height()/2, 
                   format_bytes(value, None), 
                   ha='left', va='center', fontweight='normal', fontsize=9, color='black')
    if top_n == -1 and is_cluster_mode:
        int_labels = []
        for name in display_names:
            parts = name.split('_')
            nums = [p for p in parts if p.isdigit()]
            int_labels.append(int(nums[0]) if nums else name)

        # Sort by cluster index (small to large)
        paired = sorted(zip(int_labels, reop_bytes), key=lambda x: x[0])
        int_labels, reop_bytes = zip(*paired) if paired else ([], [])

        bars1 = ax1.bar(range(len(int_labels)), reop_bytes, color='steelblue', alpha=0.7, linewidth=0.5)
        ax1.set_ylabel('Reoperation Volume', fontsize=12, labelpad=10)
        ax1.set_xlabel('Cluster Index', fontsize=12)
        ax1.set_title(f'Compact Reoperation Bytes by Cluster ({op.capitalize()}){title_suffix}', 
                      fontsize=14, pad=20, fontweight='bold')
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
        ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))

        # Show ~10 evenly spaced ticks max
        n = len(int_labels)
        step = max(1, n // 10)
        tick_positions = range(0, n, step)
        ax1.set_xticks(list(tick_positions))
        ax1.set_xticklabels([int_labels[i] for i in tick_positions], fontsize=10, rotation=45, ha='right')

        for i, (bar, value) in enumerate(zip(bars1, reop_bytes)):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, height,
                    format_bytes(value, None),
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax1.grid(True, axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    
    filename = f"{output_prefix}_branch_reoperation_bytes.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    if top_n == -1 and is_cluster_mode:
        fig1.savefig(filename.replace("_branch_", "_compact_branch_"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    if top_n == -1 and is_cluster_mode: plt.close(fig1)
    print(f"Summary: {total_non_empty} active branches found. Plot saved to {filename}")

def setup_parser(parser: argparse.ArgumentParser):
    """
    Configures the command line arguments.
    """
    parser.description = "Generates Plots and Statistics for Duplicate Read/Write Events in DXT records"

    parser.add_argument(
        "log_path",
        type=str,
        help="Specify path to darshan log.",
    )
    # Optional arguments for Darshan record filtering and operation selection
    parser.add_argument(
        "--module",
        "-m",
        nargs="?",
        default="DXT_POSIX",
        choices=["DXT_POSIX", "DXT_MPIIO"], 
        help="specify the Darshan module to generate duplicate event stats for (default: %(default)s)",
    )
    parser.add_argument(
        "--op",
        "-o",
        nargs="?",
        default="read",
        choices=["read", "write"], 
        help="specify the operation to generate duplicate event stats for (default: %(default)s)",
    )
    parser.add_argument(
        "--exclude_names",
        action='append',
        help="regex patterns for file record names to exclude"
    )
    parser.add_argument(
        "--include_names",
        action='append',
        help="regex patterns for file record names to include"
    )
    # Optional arguments for statistics and plotting
    parser.add_argument(
        "--enable_statistics",
        action='store_true',
        help="Enable printing statistics for top bins and events."
    )
    parser.add_argument(
        "--top_bins",
        type=int,
        default=3,
        help="Number of top bins to report in statistics (default: 3)."
    )
    parser.add_argument(
        "--top_events",
        type=int,
        default=10,
        help="Number of top events per bin to report in statistics (default: 10)."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        help="Specify prefix of the output plots",
    )
    parser.add_argument(
        "--time_sep",
        nargs='+',
        help="Specify time seps for integral, eg. 1 10",
    )
    parser.add_argument(
        "--no_plotting",
        action='store_true',
        help="disable plotting",
    )
    # Additional arguments for mapping
    parser.add_argument(
        "--enable_mapping",
        action='store_true',
        help="Enable mapping of reread offsets to tree/branch entries (requires --tree_branch_file)",
    )
    parser.add_argument(
        "--tree_branch_file",
        type=str,
        help="Path to tree/branch data JSON file from cluster analysis script",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection(s) to use from a multi-collection tree/branch JSON: "
             "NAME, NAME1,NAME2, or 'all'. Default: the only collection in the file, "
             "or 'CollectionTree' when the file holds several. 'all' maps every collection: "
             "exact with dumps that carry true file offsets (file_ranges / RNTuple pages), "
             "but older cursor-based TTree dumps overlap and double-count rereads.",
    )
    parser.add_argument(
        "--sort_by_time",
        action="store_true",
        help="Stable-sort the DXT segments by start_time before the analysis. Rereads are "
             "credited by row order, so this makes 'earlier read' mean earlier in time "
             "when the log lists segments out of time order (a warning is printed "
             "when it does; default: keep the log order).",
    )
    parser.add_argument(
        "--name_width",
        type=int,
        default=70,
        help="Maximum width of branch/field names in the printed summaries and "
             "branch plots. Long names keep their start and end ('Head...Tail'), "
             "and names that would still print identically are shown with more "
             "characters until they differ. 0 = never shorten (default: %(default)s).",
    )
    parser.add_argument(
        "--top_n_plot",
        type=int,
        default=-1,
        help="Number of top branches/Fields to plot (default (all): -1)."
    )

def main(args: Union[Any, None] = None):
    """
    Generates Plots and Statistics for Duplicate Read/Write Events in DXT records 
    """
    if args is None:
        parser = argparse.ArgumentParser(description="")
        setup_parser(parser)
        args = parser.parse_args()
    log_path = args.log_path
    filter_patterns=None
    filter_mode="exclude"
    if args.exclude_names and args.include_names:
        print('Error: only one of --exclude_names and --include_names may be used.')
        sys.exit(1)
    elif args.exclude_names:
        filter_patterns = args.exclude_names
        filter_mode = "exclude"
    elif args.include_names:
        filter_patterns = args.include_names
        filter_mode = "include"
    mod = args.module
    op  = args.op
    global NAME_WIDTH
    NAME_WIDTH = getattr(args, 'name_width', NAME_WIDTH)
    
    # Load tree/branch data if mapping is enabled
    tree_branch_df = pd.DataFrame()
    if args.enable_mapping:
        if not args.tree_branch_file:
            print("Error: --tree_branch_file must be specified when --enable_mapping is used.")
            sys.exit(1)
        
        tree_branch_df = load_tree_branch_data(args.tree_branch_file,
                                               collection=args.collection)
        if tree_branch_df.empty:
            print("Warning: Could not load tree/branch data. Mapping will be skipped.")
        else:
            print(f"Loaded {len(tree_branch_df)} tree/branch entries for mapping.")

    report = darshan.DarshanReport(log_path, read_all=True, filter_patterns=filter_patterns, filter_mode=filter_mode)
    if mod not in report.records:
        print(f"Error: Module '{mod}' not found in the Darshan log.\n"
            f"Please make sure that you requested one of the supported modules: DXT_POSIX or DXT_MPIIO and the log contains the requested module.\n"
            f"Available modules in this log: {list(report.records.keys())}\n")
        sys.exit(1)
    dict_list = report.records[mod].to_df()
    if len(dict_list) == 0:
        _names = getattr(report, "name_records", {}) or {}
        _names = sorted(set(str(n) for n in (_names.values() if hasattr(_names, "values") else _names)))
        print(f"Error: no {mod} records are left after name filtering "
              f"(--include_names {args.include_names}, --exclude_names {args.exclude_names}). "
              "Patterns are regular expressions matched against the file name stored in the log.")
        if _names:
            print(f"File names in this log ({len(_names)}; first 30):")
            for _n in _names[:30]:
                print(f"  {_n}")
        sys.exit(1)
    seg_key = op + "_segments"
    for idx, _dict in enumerate(dict_list):
        seg_df = order_by_time(_dict[seg_key], sort=getattr(args, 'sort_by_time', False))
        if not args.output_prefix:
            output_prefix = str(_dict.get('name', f"record_{idx}")).replace('/', '_').replace(' ', '_')
        else:
            output_prefix = args.output_prefix
            if len(dict_list) > 1:
                output_prefix = f"{output_prefix}_record{idx}"
        if seg_df.size:
            seg_df, global_reop_offsets, offset_to_bytes = compute_reoperation_bytes(seg_df)
            
            offset_mapping = {}
            if args.enable_mapping and not tree_branch_df.empty and global_reop_offsets:
                print(f"\n--- Mapping {len(global_reop_offsets)} reread offsets to tree/branch entries ---")
                offset_mapping = map_offsets_to_tree_branches(offset_to_bytes, tree_branch_df)
                generate_mapping_report(offset_mapping, format_bytes, output_prefix=output_prefix, op=op, tree_branch_df=tree_branch_df)
            
            hist_data = prepare_histogram_data(seg_df)
            if hist_data:
                if not args.no_plotting:
                    plot_offset_histogram(hist_data, output_prefix=output_prefix, op=op)
                if args.enable_statistics:
                    report_top_bins_statistics(hist_data, seg_df, format_bytes, top_bins=args.top_bins, top_events=args.top_events, op=op)
            if not args.no_plotting:
                plot_reoperation_offsets_over_time(seg_df, output_prefix=output_prefix, op=op)
            if args.enable_mapping and offset_mapping:
                plot_branch_reoperation_bytes(
                    offset_mapping,
                    format_bytes,
                    output_prefix=output_prefix+"_per_branch" if "per_branch" in args.tree_branch_file else output_prefix,
                    op=op,
                    tree_branch_df=tree_branch_df,
                    top_n=args.top_n_plot,
                    physical_reop_bytes=int(seg_df['reoperation'].sum()),
                    create_plot=not args.no_plotting,
                )

            print(seg_df)
            print(args.time_sep)
            if args.time_sep:
                time_edges = [0] + [float(t) for t in args.time_sep] + [seg_df['end_time'].max()]
                seg_df['interval'] = pd.cut(seg_df['start_time'], bins=time_edges, right=False)
                results = seg_df.groupby('interval')['reoperation'].sum()
                time_intervals = list(zip(time_edges[:-1], time_edges[1:]))
                paired_results = list(zip(time_intervals, results))
                print(paired_results)

            print(f"total overlapped bytes for {op}: ", seg_df['reoperation'].sum())
        else:
            print(f"No data found for '{op}' operation in record '{output_prefix}'. Skipping.")


if __name__ == "__main__":
    main()