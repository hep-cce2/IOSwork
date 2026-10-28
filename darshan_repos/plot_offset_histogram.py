import sys
import argparse
import bisect
import math
from typing import Any, Union
import json
import os
import io
from contextlib import redirect_stdout, redirect_stderr, ExitStack
from concurrent.futures import ProcessPoolExecutor

import darshan

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

import pandas as pd
import numpy as np

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


DEFAULT_COLLECTION = "CollectionTree"


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
    
def compute_reoperation_bytes(df, collect_chunks=False):
    """
    Optimized version using binary search and sorted interval tracking.
    Handles 100k+ rows efficiently.
    Returns both the dataframe and a mapping of offsets to their total reoperation bytes
    along with the specific end-boundaries of those overlaps.

    With collect_chunks=True every single overlap (one reread chunk) is also
    recorded in df.attrs['reop_chunks'] as numpy arrays:
        time  - start_time of the event doing the reread
        tid   - pthread_id of that event (zeros if the log has no pthread_id)
        start - first byte of the overlap
        end   - one past the last byte of the overlap
    The return value is unchanged, so existing callers keep working.
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

    if collect_chunks:
        tids = (df['pthread_id'].values if 'pthread_id' in df.columns
                else np.zeros(len(df), dtype=np.int64))
        ch_time, ch_tid, ch_start, ch_end = [], [], [], []

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

            if collect_chunks:
                ch_time.append(start_times[i])
                ch_tid.append(tids[i])
                ch_start.append(ov_start)
                ch_end.append(ov_end)

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

    if collect_chunks:
        df.attrs['reop_chunks'] = {
            'time':  np.asarray(ch_time, dtype=float),
            'tid':   np.asarray(ch_tid, dtype=tids.dtype),
            'start': np.asarray(ch_start, dtype=np.int64),
            'end':   np.asarray(ch_end, dtype=np.int64),
        }

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
    # from the exact (float) attributions and only rounded for display: truncating each
    # branch with int() dropped up to 1 byte per branch from the total.
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
    # Optional arguments for specifying non-default darshan paths
    parser.add_argument(
        "--darshan-util-path",
        type=str,
        default=None,
        help="Path to a non-default darshan-util Python installation "
             "(e.g. /path/to/darshan/lib/python3.x/site-packages, or a "
             "darshan-util build's install prefix). Prepended to sys.path "
             "before importing the darshan module. Can also be set via the "
             "DARSHAN_PYTHON_PATH environment variable.",
    )
    parser.add_argument(
        "--darshan-runtime-path",
        type=str,
        default=None,
        help="Path to a non-default darshan-util shared library install "
             "(sets LD_LIBRARY_PATH / DYLD_LIBRARY_PATH so the darshan "
             "Python bindings can find libdarshan-util). Only needed if "
             "darshan-util itself is not on the default library path.",
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
        "--no_combined_analysis",
        action="store_true",
        help="In --pthread_mode per_thread, skip the full combined (all threads) analysis "
             "and plots; only the combined total is computed (default: run it, "
             "output files <prefix>_combined_*).",
    )
    parser.add_argument(
        "--top_n_plot",
        type=int,
        default=-1,
        help="Number of top branches/Fields to plot (default (all): -1)."
    )

    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Unified main log path. In per_thread mode defaults to "
             "<output_prefix>_main.log (or <input_log_stem>_main.log). "
             "All pthread logs, summaries, and heatmap output go here in order.",
    )

    parser.add_argument(
        "--pthread_workers", type=int, default=0,
        help="Worker processes for per_thread mode (0 = automatic, 1 = sequential). "
             "Worker log output is replayed in pthread order (default: %(default)s).",
    )

    # ---- per-pthread analysis (new DXT format with pthread_id) ----
    parser.add_argument(
        "--pthread_mode",
        choices=["combined", "per_thread"],
        default="combined",
        help="combined: a segment is a reread if ANY earlier segment covered the "
             "bytes (original behavior). per_thread: rereads are detected "
             "independently inside each pthread_id, i.e. only same-thread "
             "rereads are counted, and plots/reports are written per thread "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--pthread_ids",
        type=int,
        nargs="+",
        default=None,
        help="Only analyze these pthread ids (per_thread mode and the heatmap). "
             "Default: all threads.",
    )

    # ---- pthread heatmap ----
    parser.add_argument(
        "--pthread_heatmap",
        action='store_true',
        help="Write a summary heatmap: x = time, y = pthread_id, color = reread "
             "size. With --enable_mapping the top branches/clusters are "
             "highlighted. Needs a DXT log that has pthread_id.",
    )
    parser.add_argument(
        "--heatmap_bins",
        type=int,
        default=200,
        help="Number of time bins on the heatmap x axis (default: %(default)s).",
    )
    parser.add_argument(
        "--heatmap_max_threads",
        type=int,
        default=32,
        help="Show at most this many thread rows (busiest first); the rest are "
             "merged into one 'other' row. 0 = show all (default: %(default)s).",
    )
    parser.add_argument(
        "--heatmap_agg",
        choices=["sum", "max"],
        default="sum",
        help="Cell color: 'sum' of reread bytes in the cell or the 'max' single "
             "reread in the cell (default: %(default)s).",
    )
    parser.add_argument(
        "--heatmap_scope",
        choices=["split", "combined", "same_thread"],
        default="split",
        help="split: every thread gets a same-thread row (bytes that thread read "
             "before) and a cross-thread row (reread bytes that only OTHER threads "
             "had read before). combined: one row per thread, a reread is any "
             "overlap with ANY earlier read. same_thread: one row per thread, only "
             "overlaps with the same thread's earlier reads (default: %(default)s).",
    )
    parser.add_argument(
        "--heatmap_highlight",
        choices=["branch", "cluster", "none"],
        default="branch",
        help="What to rank and highlight on the heatmap. Needs --enable_mapping "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--heatmap_top_k",
        type=int,
        default=5,
        help="Number of top branches/clusters to highlight (default: %(default)s).",
    )
    parser.add_argument(
        "--heatmap_threshold",
        type=float,
        default=0.5,
        help="A cell is outlined for a top branch/cluster when that branch/cluster "
             "accounts for at least this fraction of the cell's reread bytes "
             "(default: %(default)s).",
    )

    # ---- printout / plot labels ----
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
        help="Maximum width of branch/field names in the printed summaries, "
             "heatmap legend and branch plots. Long names keep their start and "
             "end ('Head...Tail'), and names that would still print identically "
             "are shown with more characters until they differ. 0 = never "
             "shorten (default: %(default)s).",
    )


def print_pthread_inventory(dict_list, mod):
    """
    List EVERY pthread that appears in the DXT module (all file records, reads
    and writes), so the log always shows which threads exist, even when they
    are not analysed separately. Silent for logs without pthread_id.
    """
    stats = {}
    n_rec = 0
    for _dict in dict_list:
        seen_here = set()
        for kind in ("read", "write"):
            df = _dict.get(kind + "_segments")
            if df is None or not len(df) or 'pthread_id' not in df.columns:
                continue
            df = df[df['pthread_id'].notna()]
            if df.empty:
                continue
            g = df.groupby('pthread_id').agg(n=('length', 'size'), b=('length', 'sum'),
                                             t0=('start_time', 'min'), t1=('end_time', 'max'))
            for tid, r in g.iterrows():
                s = stats.setdefault(int(tid), {'files': 0, 'read_n': 0, 'write_n': 0,
                                                'read_b': 0, 'write_b': 0,
                                                't0': float('inf'), 't1': float('-inf')})
                s[kind + '_n'] += int(r['n'])
                s[kind + '_b'] += int(r['b'])
                s['t0'] = min(s['t0'], float(r['t0']))
                s['t1'] = max(s['t1'], float(r['t1']))
                if int(tid) not in seen_here:
                    s['files'] += 1
                    seen_here.add(int(tid))
        n_rec += 1
    if not stats:
        return
    print(f"\n=== All pthreads in {mod}: {len(stats)} thread(s) across {n_rec} file record(s) ===")
    print(f"{'pthread_id':>20s} {'files':>6s} {'read segs':>10s} {'write segs':>11s} "
          f"{'read bytes':>12s} {'write bytes':>12s} {'first (s)':>10s} {'last (s)':>10s}")
    for tid, s in sorted(stats.items(), key=lambda kv: (-(kv[1]['read_n'] + kv[1]['write_n']), kv[0])):
        print(f"{tid:>20d} {s['files']:>6d} {s['read_n']:>10d} {s['write_n']:>11d} "
              f"{format_bytes(s['read_b'], None):>12s} {format_bytes(s['write_b'], None):>12s} "
              f"{s['t0']:>10.3f} {s['t1']:>10.3f}")
    print("=" * 40)


def split_by_pthread(seg_df, thread_ids=None):
    """
    Split a DXT segment DataFrame into {pthread_id: DataFrame}.
    Row order inside each thread is preserved (the interval tracking in
    compute_reoperation_bytes depends on the original event order).
    Returns None if the log has no pthread_id column (older DXT format).
    """
    if 'pthread_id' not in seg_df.columns or seg_df['pthread_id'].isna().all():
        return None
    wanted = set(int(t) for t in thread_ids) if thread_ids else None
    groups = {}
    for tid, g in seg_df.groupby('pthread_id', sort=True):
        tid = int(tid)
        if wanted is not None and tid not in wanted:
            continue
        groups[tid] = g.copy()
    return groups


def process_segments(seg_df, output_prefix, args, tree_branch_df, op, collect_chunks=False):
    """Run the full reread pipeline on one set of segments. Returns total reread bytes."""
    seg_df, global_reop_offsets, offset_to_bytes = compute_reoperation_bytes(seg_df, collect_chunks=collect_chunks)

    offset_mapping = {}
    if args.enable_mapping and not tree_branch_df.empty and global_reop_offsets:
        print(f"\n--- Mapping {len(global_reop_offsets)} reread offsets to tree/branch entries ---")
        offset_mapping = map_offsets_to_tree_branches(offset_to_bytes, tree_branch_df)
        generate_mapping_report(offset_mapping, format_bytes, output_prefix=output_prefix,
                                op=op, tree_branch_df=tree_branch_df)

    hist_data = prepare_histogram_data(seg_df)
    if hist_data:
        if not args.no_plotting:
            plot_offset_histogram(hist_data, output_prefix=output_prefix, op=op)
        if args.enable_statistics:
            report_top_bins_statistics(hist_data, seg_df, format_bytes,
                                       top_bins=args.top_bins, top_events=args.top_events, op=op)
    if not args.no_plotting:
        plot_reoperation_offsets_over_time(seg_df, output_prefix=output_prefix, op=op)
    if args.enable_mapping and offset_mapping:
        plot_branch_reoperation_bytes(
            offset_mapping,
            format_bytes,
            output_prefix=output_prefix + "_per_branch" if "per_branch" in args.tree_branch_file else output_prefix,
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
        print(list(zip(time_intervals, results)))

    total = int(seg_df['reoperation'].sum())
    print(f"total overlapped bytes for {op}: ", total)
    return total


# One read-only reference per worker rather than serializing a 250k+ row
# ROOT mapping DataFrame with every pthread task.
_THREAD_ARGS = None
_THREAD_TREE_DF = None
_THREAD_OP = None
_THREAD_PREFIX = None


def _init_pthread_worker(args, tree_branch_df, op, prefix, name_width):
    global _THREAD_ARGS, _THREAD_TREE_DF, _THREAD_OP, _THREAD_PREFIX, NAME_WIDTH
    _THREAD_ARGS = args
    _THREAD_TREE_DF = tree_branch_df
    _THREAD_OP = op
    _THREAD_PREFIX = prefix
    NAME_WIDTH = name_width


def _process_pthread_worker(task):
    """Return one thread's statistics and buffered log, with no interleaved output."""
    tid, tdf = task
    captured = io.StringIO()
    with redirect_stdout(captured), redirect_stderr(captured):
        print(f"\n########## pthread_id {tid}: {len(tdf)} {_THREAD_OP} segments ##########")
        total = process_segments(tdf, f"{_THREAD_PREFIX}_tid{tid}",
                                 _THREAD_ARGS, _THREAD_TREE_DF, _THREAD_OP)
    return tid, int(len(tdf)), total, captured.getvalue()


def print_pthread_summary(op, per_thread_stats, combined_total):
    """Table of same-thread rereads per pthread + cross-thread remainder."""
    same_thread_total = sum(s['reop'] for s in per_thread_stats.values())
    print(f"\n=== Per-pthread {op} reread summary ===")
    print(f"{'pthread_id':>22s} {'segments':>10s} {'reread bytes':>14s} {'% same-thread':>14s}")
    for tid, s in sorted(per_thread_stats.items(), key=lambda kv: kv[1]['reop'], reverse=True):
        pct = 100.0 * s['reop'] / same_thread_total if same_thread_total else 0.0
        print(f"{tid:>22d} {s['segments']:>10d} {format_bytes(s['reop'], None):>14s} {pct:>13.1f}%")
    print(f"Same-thread rereads (sum of threads): {format_bytes(same_thread_total, None)}")
    if combined_total is not None:
        cross = max(0, combined_total - same_thread_total)
        print(f"Combined rereads (all threads)      : {format_bytes(combined_total, None)}")
        print(f"Cross-thread-only rereads           : {format_bytes(cross, None)}")
    print("Definitions: same-thread = bytes a thread re-read that it had read itself earlier;")
    print("  cross-thread = bytes re-read that only OTHER threads had read earlier; combined = overlap")
    print("  with any earlier read by any thread (same-thread + cross-thread).")
    print("=" * 40)

    summary = {
        "operation": op,
        "same_thread_total_bytes": int(same_thread_total),
        "combined_total_bytes": None if combined_total is None else int(combined_total),
        "cross_thread_only_bytes": None if combined_total is None else int(max(0, combined_total - same_thread_total)),
        "threads": {str(t): {"segments": s['segments'], "reoperation_bytes": int(s['reop'])}
                    for t, s in per_thread_stats.items()},
    }
    return summary


# ===========================================================================
# pthread summary heatmap
#   x = time, y = pthread_id, color = reread size per cell.
#   Top branches / clusters (from the tree/branch mapping) are highlighted by
#   outlining the cells they dominate and by stacking the marginal bars.
# ===========================================================================

def _short_thread_labels(ids):
    """Shortest '…suffix' labels that are still unique across the given ids."""
    full = [str(int(i)) for i in ids]
    for n in range(4, 26):
        lab = ["…" + s[-n:] if len(s) > n else s for s in full]
        if len(set(lab)) == len(lab):
            return lab
    return full


def _covered_overlaps(covered, start, end):
    """Overlaps of [start, end) with a sorted list of merged intervals."""
    out = []
    idx = max(bisect.bisect_left(covered, (start,)) - 1, 0)
    while idx < len(covered):
        s, e = covered[idx]
        if s >= end:
            break
        if e <= start:
            idx += 1
            continue
        out.append((max(start, s), min(end, e)))
        idx += 1
    return out


def _covered_insert(covered, start, end):
    """Insert [start, end) into a sorted list of merged intervals."""
    new_int = (start, end)
    pos = bisect.bisect_left(covered, new_int)
    if pos > 0 and covered[pos - 1][1] >= start:
        new_int = (covered[pos - 1][0], max(covered[pos - 1][1], end))
        covered.pop(pos - 1)
        pos -= 1
    while pos < len(covered) and covered[pos][0] <= new_int[1]:
        new_int = (new_int[0], max(new_int[1], covered[pos][1]))
        covered.pop(pos)
    covered.insert(pos, new_int)


def _subtract_intervals(a, b):
    """a minus b for two sorted lists of disjoint intervals."""
    out, j = [], 0
    for s, e in a:
        while j < len(b) and b[j][1] <= s:
            j += 1
        cur, k = s, j
        while k < len(b) and b[k][0] < e:
            bs, be = b[k]
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


# ----------------------------------------------------------------------------
# TODO (not implemented, untested): reread multiplicity / source thread.
#
# Today every reread byte is counted ONCE per later read and classified
# same/cross (same wins when the issuing thread read it before). Not recorded:
#   * multiplicity: how many earlier reads covered each reread byte
#   * source threads: WHICH other threads had read it (for cross bytes)
#
# Sketch: replay events in time order and keep, per byte range, a read count and
# a set of reader thread ids, instead of plain merged intervals, e.g.
#
#   from collections import defaultdict
#
#   def compute_reread_sources(seg_df):
#       """Yield (time, reader_tid, start, end, n_prior_reads, source_tids)."""
#       # pieces: sorted, non-overlapping [start, end) with metadata
#       #   pieces[i] = (start, end, count, frozenset(reader_tids))
#       pieces = []
#       for s, e, tid, t in zip(starts, ends, tids, times):
#           for a, b, cnt, srcs in _pieces_overlapping(pieces, s, e):   # TODO helper
#               yield (t, tid, a, b, cnt, srcs - {tid} or srcs)         # cross sources
#           pieces = _pieces_insert(pieces, s, e, tid)   # TODO: split pieces, cnt += 1,
#                                                        #       srcs |= {tid}
#
# Output idea (JSON): per (branch, reader_tid, source_tid): bytes, plus a
# histogram {n_prior_reads: bytes}. Cost: pieces fragment at every distinct
# read boundary, so memory/time grow with the number of distinct boundaries
# (much larger than the merged-interval lists used now) -> put behind a flag
# such as --reread_sources.
# ----------------------------------------------------------------------------

def compute_split_reread_chunks(seg_df):
    """
    One pass over the events in order, tracking the bytes read so far by ALL
    threads and by each thread separately. For every event the reread bytes
    (overlap with anything read before) are split into
        kind 0 'same'  : bytes this thread had already read itself
        kind 1 'cross' : bytes that ONLY other threads had read before
    so same + cross == the combined reread of compute_reoperation_bytes.
    Returns chunk arrays {'time','tid','start','end','kind'}.
    """
    starts = seg_df['offset'].values.astype(np.int64)
    ends   = starts + seg_df['length'].values.astype(np.int64)
    times  = seg_df['start_time'].values
    tids   = seg_df['pthread_id'].fillna(0).values
    cov_all, cov_thr = [], {}
    ch_time, ch_tid, ch_start, ch_end, ch_kind = [], [], [], [], []
    for i in range(len(starts)):
        s, e = int(starts[i]), int(ends[i])
        cov_t = cov_thr.setdefault(int(tids[i]), [])
        allo = _covered_overlaps(cov_all, s, e)
        if allo:
            samo = _covered_overlaps(cov_t, s, e)
            crosso = _subtract_intervals(allo, samo) if samo else allo
            for kind, lst in ((0, samo), (1, crosso)):
                for a, b in lst:
                    ch_time.append(times[i]); ch_tid.append(tids[i])
                    ch_start.append(a); ch_end.append(b); ch_kind.append(kind)
        _covered_insert(cov_all, s, e)
        _covered_insert(cov_t, s, e)
    return {
        'time':  np.asarray(ch_time, dtype=float),
        'tid':   np.asarray(ch_tid, dtype=tids.dtype),
        'start': np.asarray(ch_start, dtype=np.int64),
        'end':   np.asarray(ch_end, dtype=np.int64),
        'kind':  np.asarray(ch_kind, dtype=np.int64),
    }


def collect_reread_chunks(seg_df, scope="combined", thread_ids=None):
    """
    Return the per-reread chunk arrays {'time','tid','start','end'}.
      split        : like combined, but every reread chunk is tagged 'same'
                    (the thread read those bytes before) or 'cross' (only other
                    threads read them before); same + cross == combined.
      combined    : overlap with ANY earlier read, attributed to the thread
                    that issued the later read.
      same_thread : overlap only with earlier reads of the same thread.
    Reuses the chunks recorded by a previous compute_reoperation_bytes(...,
    collect_chunks=True) call when available, otherwise recomputes them.
    """
    if scope == "split":
        chunks = compute_split_reread_chunks(seg_df)
        if thread_ids:
            keep = np.isin(chunks['tid'], np.asarray(list(thread_ids), dtype=chunks['tid'].dtype))
            chunks = {k: v[keep] for k, v in chunks.items()}
        return chunks
    if scope == "same_thread":
        groups = split_by_pthread(seg_df, thread_ids) or {}
        parts = []
        for tid, g in groups.items():
            g, _, _ = compute_reoperation_bytes(g, collect_chunks=True)
            parts.append(g.attrs['reop_chunks'])
        if not parts:
            return None
        chunks = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    else:
        chunks = seg_df.attrs.get('reop_chunks')
        if chunks is None:
            tmp, _, _ = compute_reoperation_bytes(seg_df.copy(), collect_chunks=True)
            chunks = tmp.attrs['reop_chunks']
        if thread_ids:
            keep = np.isin(chunks['tid'], np.asarray(list(thread_ids), dtype=chunks['tid'].dtype))
            chunks = {k: v[keep] for k, v in chunks.items()}
    return chunks


def _attribute_chunks_to_keys(cs, ce, tree_branch_df, key_col):
    """
    Intersect every reread chunk [cs, ce) with the tree/branch byte ranges.
    Same intersection rule as map_offsets_to_tree_branches: the bytes credited
    to a branch/cluster are exactly the bytes of the chunk inside its range.
    Returns (chunk_index, key_code, bytes, key_names) as flat match arrays.
    """
    starts = tree_branch_df['Start_byte'].values.astype(np.int64)
    if 'End_byte' in tree_branch_df.columns:
        ends = (tree_branch_df['End_byte']
                .fillna(tree_branch_df['Start_byte'] + tree_branch_df['Bytes'] - 1)
                + 1).values.astype(np.int64)
    else:
        ends = (tree_branch_df['Start_byte'] + tree_branch_df['Bytes']).values.astype(np.int64)

    codes, names = pd.factorize(tree_branch_df[key_col])
    codes = np.asarray(codes)
    share = (tree_branch_df['Share'].values.astype(float) if 'Share' in tree_branch_df.columns
             else np.ones(len(tree_branch_df)))
    order = np.argsort(starts, kind='stable')
    starts, ends, codes, share = starts[order], ends[order], codes[order], share[order]
    cummax_end = np.maximum.accumulate(ends)

    # candidates for chunk i are entries [lo_i, hi_i): start < ce_i and
    # (running max of end) > cs_i
    his = np.searchsorted(starts, ce, side='left')
    los = np.searchsorted(cummax_end, cs, side='right')

    m_chunk, m_key, m_bytes = [], [], []
    for i in np.nonzero(los < his)[0]:
        lo, hi = los[i], his[i]
        ov = np.minimum(ends[lo:hi], ce[i]) - np.maximum(starts[lo:hi], cs[i])
        ok = ov > 0
        n = int(ok.sum())
        if n:
            m_chunk.append(np.full(n, i, dtype=np.int64))
            m_key.append(codes[lo:hi][ok])
            m_bytes.append(ov[ok].astype(float) / share[lo:hi][ok])
    if not m_chunk:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0), list(names))
    return (np.concatenate(m_chunk), np.concatenate(m_key),
            np.concatenate(m_bytes), list(names))


def print_reread_definitions(scope="split"):
    """Plain-language definitions of the reread terms used in the log and plots."""
    print("\n  Definitions (reads are processed in time order; a 'reread' is any byte that")
    print("  was already read earlier; each reread is credited to the thread that issues the later read):")
    if scope in ("split", "per_thread"):
        print("    same-thread  : reread bytes that THIS thread had already read itself earlier.")
        print("    cross-thread : reread bytes that were read earlier ONLY by OTHER threads and never by")
        print("                   this thread (new for this thread, but redundant for the process).")
        print("    total        : same-thread + cross-thread = all bytes re-read by this thread, i.e. the")
        print("                   'combined' reread (overlap with any earlier read by any thread).")
    elif scope == "same_thread":
        print("    reread       : only overlaps with earlier reads of the SAME thread (cross-thread")
        print("                   rereads are not counted in this scope).")
    else:
        print("    reread       : overlap with an earlier read by ANY thread (same-thread + cross-thread")
        print("                   are not separated in this scope).")


def format_bytes_axis(y, pos=None):
    """Compact byte label for plot axes: 0B, 512B, 1.5KB, 128GB (never '195,312,500KB')."""
    y = float(y)
    if y == 0:
        return "0B"
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    while abs(y) >= 1024 and i < len(units) - 1:
        y /= 1024.0
        i += 1
    s = f"{y:.0f}" if (abs(y - round(y)) < 0.05 or abs(y) >= 100) else f"{y:.1f}"
    return f"{s}{units[i]}"


def _set_byte_axis(axis, set_lim, vmax, n=3):
    """
    Linear 0..vmax byte axis (axis = ax.xaxis / ax.yaxis, set_lim = ax.set_xlim /
    ax.set_ylim) with at most ~n binary-friendly ticks: 0, 128MB, 256MB, ...
    """
    axis.set_major_formatter(ticker.FuncFormatter(format_bytes_axis))
    if not np.isfinite(vmax) or vmax <= 0:
        set_lim(0, 1)
        axis.set_major_locator(ticker.FixedLocator([0]))
        return
    unit = 1024.0 ** int(np.floor(np.log(vmax) / np.log(1024.0)))
    step = unit
    for m in (2.0 ** e for e in range(-8, 12)):
        step = m * unit
        if vmax / step <= n:
            break
    set_lim(0, vmax * 1.05)
    axis.set_major_locator(ticker.FixedLocator(np.arange(0, vmax * 1.001, step)))


# outline colors avoid red/orange/yellow so they stay visible on the YlOrRd cells
_HIGHLIGHT_COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#17becf',
                     '#8c564b', '#e377c2', '#003f5c', '#4d4d4d']


def _union_coverage_fn(tree_branch_df):
    """Return F(x) = number of bytes < x covered by at least one tree/branch entry."""
    starts = tree_branch_df['Start_byte'].values.astype(np.int64)
    if 'End_byte' in tree_branch_df.columns:
        ends = (tree_branch_df['End_byte']
                .fillna(tree_branch_df['Start_byte'] + tree_branch_df['Bytes'] - 1)
                + 1).values.astype(np.int64)
    else:
        ends = (tree_branch_df['Start_byte'] + tree_branch_df['Bytes']).values.astype(np.int64)
    order = np.argsort(starts, kind='stable')
    starts, ends = starts[order], ends[order]
    us, ue = [], []
    for s_, e_ in zip(starts.tolist(), ends.tolist()):
        if e_ <= s_:
            continue
        if us and s_ <= ue[-1]:
            ue[-1] = max(ue[-1], e_)
        else:
            us.append(s_); ue.append(e_)
    us = np.asarray(us, dtype=np.int64); ue = np.asarray(ue, dtype=np.int64)
    cum = np.concatenate([[0], np.cumsum(ue - us)]) if len(us) else np.zeros(1, dtype=np.int64)

    def F(x):
        x = np.asarray(x, dtype=np.int64)
        if not len(us):
            return np.zeros(x.shape, dtype=np.int64)
        j = np.searchsorted(ue, x, side='right')          # intervals fully below x
        jj = np.minimum(j, len(us) - 1)
        part = np.where((j < len(us)) & (us[jj] < x), x - us[jj], 0)
        return cum[j] + part
    return F


def compute_branch_thread_kind_matrix(chunks, tree_branch_df, key_col='Branch'):
    """
    Reread bytes per (branch|cluster, pthread_id, kind), kind in {'same','cross'}
    (kind 'combined' when the chunks carry no same/cross split, e.g. old format).
    Uses the same chunk -> branch intersection as the heatmap.

    Returns (records, reconciliation):
      records         bytes per (branch|cluster, pthread, kind), sorted descending.
                      Bytes that lie outside every tree/branch entry are reported as
                      key '<unmapped>' so each (pthread, kind) can be reconciled.
      reconciliation  per (pthread, kind): physical (reread bytes), covered (inside some
                      entry), attributed (sum over branches), unmapped (physical-covered)
                      and double_counted (attributed-covered: bytes credited to more than
                      one entry because entry byte ranges overlap, e.g. several TTree
                      collections). attributed == physical - unmapped + double_counted.
    """
    if chunks is None or tree_branch_df is None or tree_branch_df.empty or len(chunks['time']) == 0:
        return [], []
    cs = np.asarray(chunks['start'], dtype=np.int64)
    ce = np.asarray(chunks['end'], dtype=np.int64)
    tid = np.asarray(chunks['tid']).astype(np.int64)
    split_kinds = 'kind' in chunks
    kind = np.asarray(chunks['kind'], dtype=np.int64) if split_kinds else np.zeros(len(cs), np.int64)
    kind_names = ['same', 'cross'] if split_kinds else ['combined']
    m_chunk, m_key, m_bytes, key_names = _attribute_chunks_to_keys(cs, ce, tree_branch_df, key_col)
    utid, tinv = np.unique(tid, return_inverse=True)
    nk, nt, nkind = len(key_names), len(utid), len(kind_names)
    label = "cluster" if key_col == 'Cluster' else "branch"

    code2 = tinv * nkind + kind                       # (thread, kind)
    size2 = nt * nkind
    F = _union_coverage_fn(tree_branch_df)
    phys = np.bincount(code2, weights=(ce - cs).astype(float), minlength=size2)
    cov = np.bincount(code2, weights=(F(ce) - F(cs)).astype(float), minlength=size2)
    attr = (np.bincount(code2[m_chunk], weights=m_bytes, minlength=size2)
            if m_chunk.size else np.zeros(size2))

    out = []
    if m_chunk.size:
        code = (m_key.astype(np.int64) * nt + tinv[m_chunk]) * nkind + kind[m_chunk]
        acc = np.bincount(code, weights=m_bytes, minlength=nk * nt * nkind)
        for c in np.nonzero(acc > 0)[0]:
            k, rem = divmod(int(c), nt * nkind)
            ti, ki = divmod(rem, nkind)
            name = key_names[k]
            out.append({label: (int(name) if key_col == 'Cluster' else str(name)),
                        "pthread_id": str(int(utid[ti])),
                        "kind": kind_names[ki],
                        "bytes": int(round(float(acc[c])))})
    recon = []
    for ti in range(nt):
        for ki in range(nkind):
            c = ti * nkind + ki
            if phys[c] <= 0:
                continue
            unm = int(round(phys[c] - cov[c]))
            dbl = int(round(attr[c] - cov[c]))
            if unm > 0:
                out.append({label: "<unmapped>", "pthread_id": str(int(utid[ti])),
                            "kind": kind_names[ki], "bytes": unm})
            recon.append({"pthread_id": str(int(utid[ti])), "kind": kind_names[ki],
                          "physical_bytes": int(round(phys[c])),
                          "attributed_bytes": int(round(attr[c])),
                          "unmapped_bytes": unm,
                          "double_counted_bytes": max(dbl, 0)})
    out.sort(key=lambda r: -r["bytes"])
    return out, recon


# =============================================================================
# TODO (not implemented): reread multiplicity and source thread of each reread
#
# Today every reread byte is counted ONCE per later read and classified as
# same-thread (the issuing thread read it before) or cross-thread (only other
# threads read it before). "Same" wins when both apply.
#
# Wanted additionally, per (branch, thread):
#   - multiplicity : how many earlier reads covered each reread byte
#   - source_threads: WHICH other threads had read those bytes first
#                     (e.g. T2 re-read 10 bytes of branch A that only T1 had read)
#
# Sketch only -- NOT executed, NOT tested. Replace the interval lists in
# compute_split_reread_chunks with a per-thread coverage map:
#
#   from collections import defaultdict
#
#   def compute_reread_sources(seg_df):
#       """Yield (time, reader_tid, start, end, source_tid, times_read_before)."""
#       df = seg_df.sort_values('start_time')
#       # per-thread sorted, merged intervals: tid -> [(start, end), ...]
#       per_thread = defaultdict(list)
#       # global read-count steps: sorted list of (offset, +1/-1 delta) per thread
#       for row in df.itertuples():
#           s, e, t = int(row.offset), int(row.offset + row.length), int(row.pthread_id)
#           for src_tid, ivs in per_thread.items():
#               for a, b in overlap(ivs, s, e):          # sorted-interval overlap helper
#                   yield (row.start_time, t, a, b, src_tid,
#                          read_count_before(a, b))      # count of earlier reads of [a,b)
#           per_thread[t] = merge(per_thread[t], s, e)   # add this read to the coverage
#
# Cost: roughly O(#reads x #threads x overlaps) time and one coverage list per
# thread in memory; consider a --reread_sources flag, off by default.
# Suggested output: <prefix>_reread_sources.json with records
#   {"branch": ..., "reader_pthread": ..., "source_pthread": ...,
#    "bytes": ..., "multiplicity_hist": {"1": ..., "2": ..., ">=3": ...}}
# =============================================================================


def plot_pthread_heatmap(chunks, tree_branch_df=None, output_prefix="output",
                         op="operation", nbins=200, top_k=5, highlight="branch",
                         threshold=0.5, max_threads=32, agg="sum", all_tids=None,
                         scope=None):
    """
    Summary heatmap of rereads per pthread over time, with top branches or
    clusters highlighted. Writes <prefix>_pthread_heatmap.png/.json.
    all_tids: every pthread_id present in the record, so threads that never
    reread anything still get a (empty) row and appear in the printed table.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle, Patch

    if chunks is None or len(chunks['time']) == 0:
        print(f"No reread chunks found, skipping pthread heatmap for {op}.")
        return

    t    = np.asarray(chunks['time'], dtype=float)
    tid  = np.asarray(chunks['tid'])
    cs   = np.asarray(chunks['start'], dtype=np.int64)
    ce   = np.asarray(chunks['end'], dtype=np.int64)
    size = (ce - cs).astype(float)

    # ---- time bins -------------------------------------------------------
    tmin, tmax = float(t.min()), float(t.max())
    if tmax <= tmin:
        tmax = tmin + 1.0
    edges = np.linspace(tmin, tmax, nbins + 1)
    dt    = edges[1] - edges[0]
    tb    = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, nbins - 1)

    # ---- thread rows: busiest first, the rest merged into 'other' --------
    uniq_src = tid
    if all_tids is not None and len(all_tids):
        uniq_src = np.concatenate([tid, np.asarray(all_tids).astype(tid.dtype)])
    uniq = np.unique(uniq_src)
    inv = np.searchsorted(uniq, tid)
    totals = np.bincount(inv, weights=size, minlength=len(uniq))
    order  = np.argsort(-totals, kind='stable')
    n_keep = len(uniq) if max_threads <= 0 else min(max_threads, len(uniq))
    has_other = len(uniq) > n_keep
    nrows = n_keep + (1 if has_other else 0)
    row_of_uniq = np.full(len(uniq), n_keep, dtype=np.int64)       # default: 'other'
    row_of_uniq[order[:n_keep]] = np.arange(n_keep)
    rows = row_of_uniq[inv]
    row_ids = [int(uniq[i]) for i in order[:n_keep]]
    labels = _short_thread_labels(row_ids)
    if has_other:
        labels.append(f"other ({len(uniq) - n_keep} threads)")

    # per-thread reread bytes for the printed table / JSON (all threads, incl. merged ones)
    split = 'kind' in chunks
    if split:
        kind_arr = np.asarray(chunks['kind'], dtype=np.int64)
        same_u  = np.bincount(inv, weights=size * (kind_arr == 0), minlength=len(uniq))
        cross_u = np.bincount(inv, weights=size * (kind_arr == 1), minlength=len(uniq))
    else:
        same_u, cross_u = totals, None
    thread_labels = list(labels)

    # split mode: a second block of rows (same block order) for cross-thread rereads
    nrows_t = nrows
    if split:
        rows = rows + np.asarray(chunks['kind'], dtype=np.int64) * nrows_t
        nrows = 2 * nrows_t
        labels = [f"{l} · same" for l in labels] + [f"{l} · cross" for l in labels]

    # ---- matrices --------------------------------------------------------
    ncell = nrows * nbins
    flat  = rows * nbins + tb
    S = np.bincount(flat, weights=size, minlength=ncell).reshape(nrows, nbins)   # sum of bytes
    if agg == "max":
        M = np.zeros(ncell)
        np.maximum.at(M, flat, size)
        M = M.reshape(nrows, nbins)
    else:
        M = S

    # ---- top branches / clusters ----------------------------------------
    top_keys, contrib = [], []
    if highlight != "none" and tree_branch_df is not None and not tree_branch_df.empty:
        key_col = 'Cluster' if highlight == 'cluster' else 'Branch'
        m_chunk, m_key, m_bytes, key_names = _attribute_chunks_to_keys(cs, ce, tree_branch_df, key_col)
        if m_chunk.size:
            key_tot = np.bincount(m_key, weights=m_bytes, minlength=len(key_names))
            ranked = [int(k) for k in np.argsort(-key_tot, kind='stable')[:max(top_k, 0)]
                      if key_tot[k] > 0]
            for k in ranked:
                sel = m_key == k
                c = np.bincount(flat[m_chunk[sel]], weights=m_bytes[sel],
                                minlength=ncell).reshape(nrows, nbins)
                contrib.append(c)
                name = key_names[k]
                top_keys.append({'name': f"cluster {int(name)}" if key_col == 'Cluster' else str(name),
                                 'key': name if key_col == 'Branch' else int(name),
                                 'bytes': float(key_tot[k])})
        else:
            print("Warning: no reread chunk intersects the tree/branch ranges; "
                  "nothing to highlight on the heatmap.")
    K = len(contrib)

    if K:
        if highlight == 'cluster':
            # sequential ramp ordered by cluster index, spread evenly over the
            # top-K so neighbouring clusters (e.g. 3, 4, 5) stay distinguishable
            frac = np.empty(K)
            frac[np.argsort([tk['key'] for tk in top_keys], kind='stable')] = \
                np.linspace(0.0, 1.0, K) if K > 1 else 0.5
            colors = [plt.cm.winter(float(f)) for f in frac]
        else:
            colors = [_HIGHLIGHT_COLORS[k % len(_HIGHLIGHT_COLORS)] for k in range(K)]
    else:
        colors = []

    # ---- figure ----------------------------------------------------------
    pos = M[M > 0]
    if pos.size == 0:
        print(f"Nothing to plot in the pthread heatmap for {op}.")
        return
    vmin, vmax = float(pos.min()), float(pos.max())
    if vmin >= vmax:
        vmin = vmax / 10.0
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig_h = float(min(18, max(6.5, 3.0 + 0.28 * nrows)))
    fig = plt.figure(figsize=(14, fig_h))
    gs = fig.add_gridspec(2, 3, width_ratios=[16, 2.6, 0.4], height_ratios=[2.2, 6],
                          hspace=0.05, wspace=0.05)
    ax  = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 2])
    if split:
        # one marginal per block, each with its OWN scale: cross-thread rereads are
        # often orders of magnitude smaller than same-thread ones and would
        # otherwise vanish next to them. The two right-hand axes split the column
        # in half, matching the two equal row blocks of the heatmap.
        gt = gs[0, 0].subgridspec(2, 1, hspace=0.18)
        axt_list = [fig.add_subplot(gt[i], sharex=ax) for i in range(2)]
        gr = gs[1, 1].subgridspec(2, 1, hspace=0.0)
        axr_list = [fig.add_subplot(gr[i]) for i in range(2)]
        blocks = [(slice(0, nrows_t), "same-thread"), (slice(nrows_t, nrows), "cross-thread")]
    else:
        axt_list = [fig.add_subplot(gs[0, 0], sharex=ax)]
        axr_list = [fig.add_subplot(gs[1, 1], sharey=ax)]
        blocks = [(slice(0, nrows), None)]

    mesh = ax.pcolormesh(edges, np.arange(nrows + 1), np.ma.masked_less_equal(M, 0),
                         cmap="YlOrRd", norm=norm, shading='flat')
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(nrows, 0)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels([f"{n + 1}  {lab}" for n, lab in enumerate(labels)], fontsize=9)   # row number first
    ax.set_ylabel("pthread_id")
    ax.set_xlabel("Start time (s)")

    if split:
        ax.axhline(nrows_t, color="black", linewidth=1.5)     # same-thread block above, cross-thread below

    lw = 1.6 if nbins <= 100 else 1.0
    highlighted_cells = []
    for k in range(K):
        share = np.divide(contrib[k], S, out=np.zeros_like(S), where=S > 0)
        ys, xs = np.nonzero(share >= threshold)
        highlighted_cells.append([(int(y), int(x)) for y, x in zip(ys, xs)])
        if len(ys):
            rects = [Rectangle((edges[x], y), dt, 1) for y, x in zip(ys, xs)]
            ax.add_collection(PatchCollection(rects, facecolor='none',
                                              edgecolor=colors[k], linewidth=lw))

    cbar = fig.colorbar(mesh, cax=cax,
                        label=("Largest single reread per cell" if agg == "max"
                               else "Reread bytes per cell"))
    cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=4, numticks=8))
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    cbar.ax.yaxis.set_minor_locator(ticker.NullLocator())

    # top marginal: per time bin, stacked by top branches/clusters
    centers = (edges[:-1] + edges[1:]) / 2
    rest_color = "#bbbbbb" if K else "steelblue"
    for axt_b, (sl, bname) in zip(axt_list, blocks):
        per_time_total = S[sl].sum(axis=0)
        bottom = np.zeros(nbins)
        for k in range(K):
            v = contrib[k][sl].sum(axis=0)
            axt_b.bar(centers, v, bottom=bottom, width=dt, color=colors[k], linewidth=0)
            bottom += v
        other_t = np.clip(per_time_total - bottom, 0, None)
        axt_b.bar(centers, other_t, bottom=bottom, width=dt, color=rest_color, linewidth=0)
        axt_b.set_ylabel(bname or "per time bin", fontsize=8)
        axt_b.tick_params(labelbottom=False, labelsize=8)
        _set_byte_axis(axt_b.yaxis, axt_b.set_ylim, float(per_time_total.max()), 3)
    axt_list[0].set_title(f"Reread size per pthread over time ({op.capitalize()})"
                          + ("  -  same-thread rows above the line, cross-thread rows below" if split else ""),
                          fontsize=11, fontweight="bold")

    # right marginal: per thread row, stacked by top branches/clusters
    for j, (axr_b, (sl, bname)) in enumerate(zip(axr_list, blocks)):
        ypos = np.arange(sl.start, sl.stop) + 0.5
        row_total = S[sl].sum(axis=1)
        left = np.zeros(len(ypos))
        for k in range(K):
            v = contrib[k][sl].sum(axis=1)
            axr_b.barh(ypos, v, left=left, height=1, color=colors[k], linewidth=0)
            left += v
        other_r = np.clip(row_total - left, 0, None)
        axr_b.barh(ypos, other_r, left=left, height=1, color=rest_color, linewidth=0)
        axr_b.tick_params(labelleft=False, labelsize=7 if split else 8)
        _set_byte_axis(axr_b.xaxis, axr_b.set_xlim, float(row_total.max()), 2)
        if split:
            axr_b.set_ylim(sl.stop, sl.start)          # same row order/direction as the heatmap
            axr_b.text(0.97, 0.02, bname, transform=axr_b.transAxes, ha='right', va='bottom',
                       fontsize=7, color='0.2',
                       bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
            if j == 0:
                axr_b.xaxis.tick_top()                 # keep the two scales' labels apart
        else:
            axr_b.set_xlabel("per thread", fontsize=8)

    if K:
        legend_names = unique_short_names([tk['name'] for tk in top_keys],
                                          width=min(NAME_WIDTH, 48) if NAME_WIDTH > 0 else 0)
        handles = [Patch(fc="none", ec=colors[k], lw=1.8, label=legend_names[k])
                   for k in range(K)]
        handles.append(Patch(fc="#bbbbbb", label="other / unmapped"))
        ax.legend(handles=handles,
                  title=f"Top-{K} {highlight}s (outline = ≥{threshold:.0%} of the cell's reread bytes)",
                  loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=min(3, K + 1), fontsize=8, title_fontsize=8, frameon=False)

    png = f"{output_prefix}_pthread_heatmap.png"
    fig.savefig(png, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---- JSON with the binned data, for reuse ------------------------------
    # bytes per (branch|cluster, pthread_id, same|cross); entries sum to the per-thread totals
    branch_thread_kind, btk_recon = compute_branch_thread_kind_matrix(
        chunks, tree_branch_df, 'Cluster' if highlight == 'cluster' else 'Branch')
    _unm = sum(r["unmapped_bytes"] for r in btk_recon)
    _dbl = sum(r["double_counted_bytes"] for r in btk_recon)
    if btk_recon:
        print(f"  Branch attribution check: unmapped (outside every tree/branch entry) = "
              f"{format_bytes(_unm, None)}, double-counted (entries with overlapping byte ranges) = "
              f"{format_bytes(_dbl, None)}")
        if _dbl > 0:
            print("  Warning: some reread bytes are credited to more than one entry because the "
                  "selected tree/branch byte ranges overlap (TTree ranges from several collections). "
                  "Select a single collection (--collection) for exact per-branch numbers.")
    out = {
        "operation": op,
        "agg": agg,
        "threshold": threshold,
        "highlight": highlight if K else "none",
        "time_edges": [float(x) for x in edges],
        "scope": "split" if split else "single",
        "pthread_rows": ([str(i) for i in row_ids] + (["other"] if has_other else [])) * (2 if split else 1),
        "row_kind": (["same"] * nrows_t + ["cross"] * nrows_t) if split else None,
        "pthread_labels": labels,
        "all_pthread_ids": [str(int(u)) for u in uniq],
        "other_pthread_ids": [str(int(uniq[i])) for i in order[n_keep:]],
        "per_pthread_reread_bytes": [
            {"pthread_id": str(int(uniq[i])),
             "row": thread_labels[r] if r < n_keep else thread_labels[-1],
             "same_row_number": (r if r < n_keep else n_keep) + 1,
             "cross_row_number": None if not split else nrows_t + (r if r < n_keep else n_keep) + 1,
             "same_thread_bytes": int(same_u[i]),
             "cross_thread_bytes": None if cross_u is None else int(cross_u[i]),
             "total_bytes": int(totals[i])}
            for r, i in enumerate(order)
        ],
        "branch_thread_kind": branch_thread_kind,
        "branch_thread_kind_reconciliation": btk_recon,
        "cell_sum_bytes": [[int(v) for v in r] for r in S],
        "cell_value": [[int(v) for v in r] for r in M],
        "top": [
            {**top_keys[k],
             "highlighted_cells": [{"row": y, "time_bin": x} for y, x in highlighted_cells[k]]}
            for k in range(K)
        ],
    }
    with open(f"{output_prefix}_pthread_heatmap.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"pthread heatmap saved to {png}  ({nrows} rows x {nbins} time bins)")

    if K:
        names_p = unique_short_names([tk['name'] for tk in top_keys])
        col = max(len(s) for s in names_p)
        for k in range(K):
            print(f"  {k + 1}. {names_p[k]:{col}s} {format_bytes(top_keys[k]['bytes'], None):>10s}"
                  f"  outlined cells: {len(highlighted_cells[k])}")

    print_reread_definitions(scope or ("split" if split else "combined"))

    # every pthread in the record, busiest first, with the heatmap row number(s) it is drawn on
    N = nrows_t
    print(f"\n  All {len(uniq)} pthreads in this record (reread bytes, {op}):")
    print("  'row' = the row number on the heatmap, counted from the top of the plot starting at 1;")
    print("  the same number starts each y-axis label in the figure.")
    if split:
        print(f"    {'rows 1-' + str(N):<16s}: same-thread rows, one per thread (busiest thread first)")
        print(f"    {'rows ' + str(N + 1) + '-' + str(2 * N):<16s}: cross-thread rows, same thread order "
              f"(a thread on same-thread row k has its cross-thread row at k + {N})")
    else:
        print(f"    {'rows 1-' + str(N):<16s}: one row per thread (busiest thread first)")
    if has_other:
        print(f"    '*' = this thread is merged into the last row of its block, "
              f"'{thread_labels[-1]}'")
    hdr = f"    {'same row':>9s}  " + (f"{'cross row':>9s}  " if split else "") + f"{'pthread_id':>20s}  "
    hdr += (f"{'same-thread':>12s}  {'cross-thread':>12s}  {'total':>12s}" if split
            else f"{'reread':>12s}")
    print(hdr.replace("same row", "row" if not split else "same row"))
    for r, i in enumerate(order):
        ridx = r if r < n_keep else n_keep                    # 0-based row within its block
        star = "*" if r >= n_keep else ""
        line = f"    {str(ridx + 1) + star:>9s}  "
        if split:
            line += f"{str(N + ridx + 1) + star:>9s}  "
        line += f"{int(uniq[i]):>20d}  "
        if split:
            line += (f"{format_bytes(same_u[i], None):>12s}  "
                     f"{format_bytes(cross_u[i], None):>12s}  "
                     f"{format_bytes(totals[i], None):>12s}")
        else:
            line += f"{format_bytes(totals[i], None):>12s}"
        print(line)
    if has_other:
        print(f"    ({len(uniq) - n_keep} threads share the '{thread_labels[-1]}' row; raise "
              f"--heatmap_max_threads, or use 0, to give each its own row)")


def run_pthread_heatmap(seg_df, output_prefix, args, tree_branch_df, op):
    """Driver: validate the input, gather the chunks, draw the heatmap."""
    if 'pthread_id' not in seg_df.columns or seg_df['pthread_id'].isna().all():
        print(f"Warning: no 'pthread_id' in record '{output_prefix}' "
              f"(older DXT log format); skipping the pthread heatmap.")
        return
    highlight = args.heatmap_highlight
    if highlight != "none" and (not args.enable_mapping or tree_branch_df is None
                                or tree_branch_df.empty):
        print("Note: --enable_mapping with a loadable --tree_branch_file is needed "
              "to highlight branches/clusters; drawing the heatmap without highlights.")
        highlight = "none"
    chunks = collect_reread_chunks(seg_df, args.heatmap_scope, args.pthread_ids)
    all_tids = seg_df['pthread_id'].dropna().unique()
    if args.pthread_ids:
        all_tids = [t for t in all_tids if int(t) in set(int(x) for x in args.pthread_ids)]
    plot_pthread_heatmap(chunks, tree_branch_df, output_prefix=output_prefix, op=op,
                         nbins=args.heatmap_bins, top_k=args.heatmap_top_k,
                         highlight=highlight, threshold=args.heatmap_threshold,
                         max_threads=args.heatmap_max_threads, agg=args.heatmap_agg,
                         all_tids=all_tids, scope=args.heatmap_scope)


def _run_main(args: Union[Any, None] = None):
    """
    Generates Plots and Statistics for Duplicate Read/Write Events in DXT records
    """
    if args is None:
        parser = argparse.ArgumentParser(description="")
        setup_parser(parser)
        args = parser.parse_args()
    log_path = args.log_path
    filter_patterns = None
    filter_mode = "exclude"
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
    op = args.op

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

    report = darshan.DarshanReport(log_path, read_all=True,
                                   filter_patterns=filter_patterns, filter_mode=filter_mode)
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
    global NAME_WIDTH
    NAME_WIDTH = getattr(args, 'name_width', NAME_WIDTH)
    print_pthread_inventory(dict_list, mod)
    seg_key = op + "_segments"
    want_heatmap = bool(args.pthread_heatmap) and not args.no_plotting
    for idx, _dict in enumerate(dict_list):
        seg_df = order_by_time(_dict[seg_key], sort=getattr(args, 'sort_by_time', False))
        if not args.output_prefix:
            output_prefix = str(_dict.get('name', f"record_{idx}")).replace('/', '_').replace(' ', '_')
        else:
            output_prefix = args.output_prefix
            if len(dict_list) > 1:
                # one fixed prefix for several file records would make each record
                # overwrite the previous record's plots and JSON files
                output_prefix = f"{output_prefix}_record{idx}"

        if not seg_df.size:
            print(f"No data found for '{op}' operation in record '{output_prefix}'. Skipping.")
            continue

        handled = False
        if args.pthread_mode == "per_thread":
            groups = split_by_pthread(seg_df, args.pthread_ids)
            if groups is None:
                print(f"Warning: '{mod}' segments in record '{output_prefix}' have no "
                      f"'pthread_id' column (older DXT log format); falling back to combined mode.")
            elif not groups:
                print(f"No segments for the requested pthread ids in record '{output_prefix}'. Skipping.")
                continue
            else:
                # Combined total (all threads) so cross-thread rereads can be derived.
                # Only meaningful when no thread filter is applied.
                combined_total = None
                if not args.pthread_ids:
                    if getattr(args, 'no_combined_analysis', False):
                        tmp, _, _ = compute_reoperation_bytes(seg_df.copy())
                        combined_total = int(tmp['reoperation'].sum())
                    else:
                        # Full combined analysis (all threads together): mapping report,
                        # branch statistics and the usual plots, files named <prefix>_combined_*
                        print(f"\n########## ALL pthreads (combined): {len(seg_df)} {op} segments ##########")
                        combined_total = process_segments(seg_df.copy(), f"{output_prefix}_combined",
                                                          args, tree_branch_df, op)
                elif not getattr(args, 'no_combined_analysis', False):
                    print("Note: combined (all-thread) analysis is skipped because --pthread_ids "
                          "selects a subset of threads.")

                per_thread_stats = {}
                workers = getattr(args, 'pthread_workers', 0)
                if workers < 0:
                    raise ValueError("--pthread_workers must be non-negative")
                workers = min(len(groups), workers or (os.cpu_count() or 1))

                if workers == 1:
                    # Preserve original direct printing when there is no parallelism.
                    for tid, tdf in groups.items():
                        print(f"\n########## pthread_id {tid}: {len(tdf)} {op} segments ##########")
                        total = process_segments(tdf, f"{output_prefix}_tid{tid}",
                                                 args, tree_branch_df, op)
                        per_thread_stats[tid] = {"segments": int(len(tdf)), "reop": total}
                else:
                    # executor.map yields results in submission/pthread order, even if
                    # jobs finish out of order. Only the parent writes the console log.
                    # Each child uses a unique per-pthread filename prefix.
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_init_pthread_worker,
                        initargs=(args, tree_branch_df, op, output_prefix, NAME_WIDTH),
                    ) as pool:
                        for tid, nsegments, total, thread_log in pool.map(
                            _process_pthread_worker, groups.items()
                        ):
                            sys.stdout.write(thread_log)
                            sys.stdout.flush()
                            per_thread_stats[tid] = {"segments": nsegments, "reop": total}

                summary = print_pthread_summary(op, per_thread_stats, combined_total)
                with open(f"{output_prefix}_pthread_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                handled = True

        if not handled:
            # combined mode (original behavior); record the single rereads for
            # the heatmap in the same pass when it is requested
            process_segments(seg_df, output_prefix, args, tree_branch_df, op,
                             collect_chunks=(want_heatmap and args.heatmap_scope == "combined"))

        if want_heatmap:
            run_pthread_heatmap(seg_df, output_prefix, args, tree_branch_df, op)


class _TeeStream:
    """Print to the terminal and the unified log at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main(args: Union[Any, None] = None):
    """Run analysis with one organized main log (including worker output)."""
    if args is None:
        parser = argparse.ArgumentParser(description="")
        setup_parser(parser)
        args = parser.parse_args()

    log_file = getattr(args, 'log_file', None)
    if log_file is None and args.pthread_mode == 'per_thread':
        prefix = args.output_prefix or os.path.splitext(os.path.basename(args.log_path))[0]
        log_file = f"{prefix}_main.log"

    if not log_file:
        return _run_main(args)

    # Workers return their buffered text; only this parent process writes the
    # unified log. This avoids interleaving, even when workers finish out of order.
    directory = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(directory, exist_ok=True)
    with open(log_file, 'w', encoding='utf-8', buffering=1) as log_handle:
        with redirect_stdout(_TeeStream(sys.stdout, log_handle)), \
             redirect_stderr(_TeeStream(sys.stderr, log_handle)):
            print(f"Main analysis log: {log_file}")
            return _run_main(args)


if __name__ == "__main__":
    main()
