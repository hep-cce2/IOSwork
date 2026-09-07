import sys
import argparse
import bisect
from typing import Any, Union
import json

import darshan

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

import pandas as pd
import numpy as np

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


def load_tree_branch_data(tree_branch_file):
    """Load cluster/branch data from JSON. Supports TTree and RNTuple formats."""
    try:
        with open(tree_branch_file, 'r') as f:
            data = json.load(f)

        data_type = data.get("type", "TTree")
        clusters  = data.get("clusters", [])
        rows      = []

        for cluster in clusters:
            cluster_idx = cluster["cluster_index"]
            begin       = cluster["begin"]
            end         = cluster["end"]
            entries     = cluster["entries"]

            if data_type == "RNTuple" and "fields" in cluster:
                # ── RNTuple path ──────────────────────────────────────────
                for field_data in cluster["fields"]:
                    field_name   = field_data["field"]
                    column_id    = field_data.get("column_id", -1)
                    field_id     = field_data.get("field_id", -1)
                    column_type  = field_data.get("column_type", "")
                    comp_bytes   = field_data.get("compressed_bytes", 0)   # was "bytes"
                    uncomp_bytes = field_data.get("uncompressed_bytes", 0)
                    n_pages      = field_data.get("n_pages", 1)

                    pages = field_data.get("pages", [])
                    if pages:
                        # Prefer page-level granularity when available:
                        # each page becomes its own row so offset mapping
                        # can resolve to the exact page, not just the field.
                        for page in pages:
                            rows.append({
                                'Cluster':            cluster_idx,
                                'Branch':             field_name,
                                'Column_id':          column_id,
                                'Field_id':           field_id,
                                'Column_type':        column_type,
                                'Begin':              begin,
                                'End':                end,
                                'Entries':            entries,
                                'Start_byte':         page["offset"],
                                'End_byte':           page["end"] - 1,
                                'Bytes':              page["compressed_bytes"],
                                'Uncompressed_bytes': page.get("uncompressed_bytes", 0),
                                'Baskets':            1,           # one page = one unit
                                'Page_index':         page["page_index"],
                                'N_elements':         page.get("n_elements", 0),
                                'Is_compressed':      page.get("is_compressed", False),
                                'Granularity':        'page',
                            })
                    else:
                        # Fall back to field-level if pages were not written
                        rows.append({
                            'Cluster':            cluster_idx,
                            'Branch':             field_name,
                            'Column_id':          column_id,
                            'Field_id':           field_id,
                            'Column_type':        column_type,
                            'Begin':              begin,
                            'End':                end,
                            'Entries':            entries,
                            'Start_byte':         field_data.get("start_byte", 0),
                            'End_byte':           field_data.get("end_byte", 0),
                            'Bytes':              comp_bytes,
                            'Uncompressed_bytes': uncomp_bytes,
                            'Baskets':            n_pages,
                            'Page_index':         -1,
                            'N_elements':         0,
                            'Is_compressed':      False,
                            'Granularity':        'field',
                        })

            elif "branches" in cluster:
                # ── TTree path ────────────────────────────────────────────
                for branch_data in cluster["branches"]:
                    rows.append({
                        'Cluster':            cluster_idx,
                        'Branch':             branch_data["branch"],
                        'Column_id':          -1,
                        'Field_id':           -1,
                        'Column_type':        '',
                        'Begin':              begin,
                        'End':               end,
                        'Entries':            entries,
                        'Start_byte':         branch_data["start_byte"],
                        'End_byte':           branch_data["end_byte"],
                        'Bytes':              branch_data["bytes"],
                        'Uncompressed_bytes': 0,
                        'Baskets':            branch_data.get("baskets", 1),
                        'Page_index':         -1,
                        'N_elements':         0,
                        'Is_compressed':      False,
                        'Granularity':        'basket',
                    })

            else:
                # ── cluster-level fallback (no per-branch/field data) ─────
                rows.append({
                    'Cluster':            cluster_idx,
                    'Branch':             f'cluster_{begin}_{end}',
                    'Column_id':          -1,
                    'Field_id':           -1,
                    'Column_type':        '',
                    'Begin':              begin,
                    'End':                end,
                    'Entries':            entries,
                    'Start_byte':         cluster["start_byte"],
                    'End_byte':           cluster["end_byte"],
                    'Bytes':              cluster["total_bytes"],
                    'Uncompressed_bytes': cluster.get("uncompressed_bytes", 0),
                    'Baskets':            cluster.get("max_baskets",
                                          cluster.get("max_pages", 1)),
                    'Page_index':         -1,
                    'N_elements':         0,
                    'Is_compressed':      False,
                    'Granularity':        'cluster',
                })

        df = pd.DataFrame(rows)

        if not df.empty:
            print(f"Loaded DataFrame from {data_type}")
            print(f"Columns:     {list(df.columns)}")
            print(f"Shape:       {df.shape}")
            print(f"Granularity: {df['Granularity'].value_counts().to_dict()}")
            print(f"First few rows:\n{df.head()}")
        else:
            print("Warning: No data loaded from the file")

        return df

    except Exception as e:
        print(f"Error loading tree/branch data: {e}")
        import traceback
        traceback.print_exc()
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

    for offset in offsets:
        info = reop_data[offset]
        if isinstance(info, dict) and 'bytes' in info:
            total_reop_bytes = info['bytes']
            ov_end = info['end']
        else:
            # Fallback if old structure
            total_reop_bytes = info
            ov_end = offset + 1

        ov_start = offset
        matches  = []
        
        # Vectorized check: does the reread interval intersect the branch?
        in_range = ((start_bytes < ov_end) & ((end_bytes) > ov_start))
        branch_indices = np.where(in_range)[0]

        for branch_idx in branch_indices:
            row        = tree_branch_df.iloc[branch_idx]
            b_start    = start_bytes[branch_idx]
            b_end      = end_bytes[branch_idx] # This is start + length

            # Calculate exact byte intersection
            intersect_start = max(ov_start, b_start)
            intersect_end   = min(ov_end, b_end)
            intersect_len   = max(0, intersect_end - intersect_start)

            # Attribute reoperation penalty proportionally to intersection size
            ratio = intersect_len / (ov_end - ov_start) if (ov_end - ov_start) > 0 else 0
            attributed_reop_bytes = total_reop_bytes * ratio

            match = {
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
    for i, (branch, count, reop_bytes) in enumerate(sorted_branches[:10]):
        temporal = analyze_temporal_distribution(
            branch_cluster_indices[branch], total_clusters)
        display = branch if len(branch) <= 50 else branch[:47] + "..."
        print(f"  {i+1:2d}. {display:50s} : "
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
                offset_to_total_bytes[ov_start] = {'bytes': 0, 'end': ov_end}
            offset_to_total_bytes[ov_start]['bytes'] += overlap_amount
            offset_to_total_bytes[ov_start]['end'] = max(offset_to_total_bytes[ov_start]['end'], ov_end)
            
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


def plot_branch_reoperation_bytes(offset_mapping, format_bytes, output_prefix="output", op="operation", top_n=-1, tree_branch_df=None): 
    """
    Generates and saves a bar plot of reoperation bytes per branch/field.
    Uses corrected intersection attribution logic.
    """
    total_all_branches = 0
    if tree_branch_df is not None and not tree_branch_df.empty:
        total_all_branches = len(tree_branch_df['Branch'].unique())
    
    branch_reoperation_bytes = {}
    
    # Safely aggregate the newly calculated volumetric reoperation_bytes from matches
    for offset, matches in offset_mapping.items():
        if matches:
            for match in matches:
                branch_name = str(match['branch'])
                branch_reoperation_bytes[branch_name] = branch_reoperation_bytes.get(branch_name, 0) + match.get('reoperation_bytes', 0)

    if not branch_reoperation_bytes:
        print(f"No branch reoperation data found to plot for {op}.")
        return
    
    # 1. Sort and Filter
    non_empty_branches = [(name, bytes_val) for name, bytes_val in branch_reoperation_bytes.items() if bytes_val > 0]
    total_non_empty = len(non_empty_branches)
    
    is_cluster_mode = non_empty_branches[0][0].startswith("cluster_")

    if is_cluster_mode:
        non_empty_branches.sort(key=lambda x: int(x[0].split('_')[1]))

    if top_n != -1 or not is_cluster_mode:
        non_empty_branches.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Apply Top N Slicing
    if top_n != -1:
        branches_to_plot = non_empty_branches[:top_n]
    else:
        branches_to_plot = non_empty_branches

    if is_cluster_mode:
        branches_to_plot.sort(key=lambda x: int(x[0].split('_')[1]))
    
    if not branches_to_plot:
        return

    # 3. Prepare for Horizontal Bar Chart
    branches_to_plot.reverse()
    
    branch_names = [item[0] for item in branches_to_plot]
    reop_bytes = [item[1] for item in branches_to_plot]
    display_names = [n[:50] + "..." if len(n) > 50 else n for n in branch_names]
    
    fig_height = max(2.5, 1 + 0.3 * len(display_names))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    if top_n == -1 and is_cluster_mode:
        fig1, ax1 = plt.subplots(figsize=(14, 4))
    
    bars = ax.barh(range(len(display_names)), reop_bytes, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel('Reoperation Volume', fontsize=12, labelpad=10)
    ax.set_ylabel('Cluster Index' if is_cluster_mode else 'Branch/Field Names', fontsize=12)
    
    title_suffix = f" - Top {len(branches_to_plot)}" if top_n != -1 else " (All Active)"
    ax.set_title(f'Reoperation Bytes by {"Cluster" if is_cluster_mode else "Branch"} ({op.capitalize()}){title_suffix}', 
                 fontsize=10, pad=10, fontweight='bold')
    
    # Nicer ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.grid(True, axis='x', which='major', linestyle='-', alpha=0.4)
    ax.grid(True, axis='x', which='minor', linestyle=':', alpha=0.2)
    
    max_width = max(reop_bytes) if reop_bytes else 0
    for bar, value in zip(bars, reop_bytes):
        width = bar.get_width()
        if width > max_width * 0.15:
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
    
    stats = {
        "operation": op,
        "total_branches": total_all_branches,
        "non_zero_branches": total_non_empty,
        "total_reop_bytes": int(sum(reop_bytes)),
        "branches": [{"name": n, "bytes": b} for n, b in non_empty_branches[::-1]] 
    }
    with open(f"{output_prefix}_branch_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

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
        "--top_n_plot",
        type=int,
        default=-1,
        help="Number of top branches/Fields to plot (default (all): -1)."
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
    parser.add_argument(
        "--tree_branch_file",
        type=str,
        help="Path to tree/branch data JSON file from cluster analysis script",
    )
    parser.add_argument(
        "--enable_mapping",
        action='store_true',
        help="Enable mapping of reread offsets to tree/branch entries (requires --tree_branch_file)",
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
    
    # Load tree/branch data if mapping is enabled
    tree_branch_df = pd.DataFrame()
    if args.enable_mapping:
        if not args.tree_branch_file:
            print("Error: --tree_branch_file must be specified when --enable_mapping is used.")
            sys.exit(1)
        
        tree_branch_df = load_tree_branch_data(args.tree_branch_file)
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
    seg_key = op + "_segments"
    for idx, _dict in enumerate(dict_list):
        seg_df = _dict[seg_key]
        if not args.output_prefix:
            output_prefix = str(_dict.get('name', f"record_{idx}")).replace('/', '_').replace(' ', '_')
        else:
            output_prefix = args.output_prefix
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
                        top_n=args.top_n_plot
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