# ROOT cluster dumper

Dumps the cluster layout and the on-disk data volume of every TTree and RNTuple in a ROOT file as
JSON. The Darshan reread analysis (`plot_offset_histogram.py`) reads this JSON to map file offsets
from DXT traces to branches and fields.

Script: `dump_cluster_boundary.py` (source: `root_cluster_dumper.py`). Needs PyROOT (tested layout
against ROOT 6.36.04).

## Usage

```
python dump_cluster_boundary.py input.root NAME[,NAME2,...]|all [max_workers] [per_branch=0/1] [output.json]
```

| Argument | Meaning |
|---|---|
| `input.root` | file to dump |
| `NAME...` or `all` | one collection, a comma list, or every TTree/RNTuple key in the file |
| `max_workers` | accepted for compatibility, currently unused |
| `per_branch` | `1` adds the per-branch / per-field / per-page detail (needed for the Darshan mapping) |
| `output.json` | output path; stdout when omitted |

Example (all collections, per-branch detail):

```
python dump_cluster_boundary.py file.root all 16 1 test_dump_per_branch_file.json
```

One explicitly named collection writes the original single-collection shape. `all` or a comma list
writes `{"file_name": ..., "collections": {name: dump, ...}, "errors": {...}}`.

## Which offsets are real file positions

This is the most important thing to know when using the output.

| Where | Field | Meaning |
|---|---|---|
| TTree | `branches[].file_ranges` | **True** `[start, end)` file positions of the baskets of that branch in that cluster (adjacent baskets merged). Use these. |
| TTree | `start_byte`, `end_byte` (cluster, branch, `branch_totals`) | Running **per-tree cursors**, not file positions. Every tree starts at 0, so they overlap between collections. Kept for old consumers only. |
| TTree | `branch_totals[].file_start`, `file_end` | Lowest start / highest end of the branch's `file_ranges`. |
| RNTuple | `pages[].offset`, `end` | **True** file positions of each page. |
| RNTuple | `start_byte`, `end_byte` (cluster, field, `field_totals`) | Running cursors, not file positions. |

Consumers that map reads to branches must use `file_ranges` / page offsets. With the old cursors,
rereads are credited to every collection whose cursor range overlaps, which double counts (this was
found on an ATLAS EVNT file: 9 TTree collections all starting at 0).

## TTree output

```json
{
  "type": "TTree", "tree_name": "CollectionTree", "total_entries": 5000,
  "clusters": [{
    "cluster_index": 0, "begin": 0, "end": 1332, "entries": 1332,
    "start_byte": 0, "end_byte": 29962556, "total_bytes": 29962557, "max_baskets": 1341,
    "branches": [{
      "branch": "EventInfo_p4_McEventInfo",
      "start_byte": 0, "end_byte": 11974, "bytes": 11975, "baskets": 9,
      "file_ranges": [[275457, 276735], [2537057, 2538666]]
    }]
  }],
  "offsets": "start_byte/end_byte are per-tree cursors; branches[].file_ranges are file positions",
  "branch_totals": {"EventInfo_p4_McEventInfo": {"entries": 5000, "bytes": 42169, "baskets": 33,
      "start_byte": 0, "end_byte": 89912926, "file_start": 275457, "file_end": 40569867}}
}
```

- `bytes` / `total_bytes` are compressed on-disk basket sizes (including the basket header).
- Branch names are `GetFullName()`; sub-branches are included as their own entries.
- Every basket is assigned to the cluster that contains its **first entry**. A basket that starts
  inside a cluster is never dropped, even when baskets do not align with cluster boundaries.
- The number of baskets is `GetWriteBasket()`, not `GetMaxBaskets()` (see "Fixed bugs").

## RNTuple output

Same shape as the TTree output: `cluster.branches[]` becomes `cluster.fields[]`, `max_baskets`
becomes `max_pages`, `branch_totals` becomes `field_totals`.

Each field entry (one per physical column) lists its pages:

```json
{"page_index": 0, "offset": 3165131757, "end": 3165508634, "compressed_bytes": 376877,
 "uncompressed_bytes": "...", "n_elements": 110785, "is_compressed": true,
 "virtual": false, "shared_count": 2, "is_owner": true,
 "shared_with": ["JetETMissNeutralParticleFlowObjectsAux:.eta._0"]}
```

(Offsets, sizes and field names are from a real AOD page shared by two fields;
`uncompressed_bytes` is elided here.)

### Logical versus physical bytes

The RNTuple writer can store identical pages once and let several columns reference them (same
`offset` and size). The dump therefore carries two volumes:

| Key | Meaning |
|---|---|
| `compressed_bytes` (top level), `total_bytes` (cluster) | **Logical**: summed per column, a shared page is counted once per column. Equals `RNTupleInspector.GetCompressedSize()`. |
| `physical_compressed_bytes`, cluster `physical_bytes` | **Physical**: each distinct page counted once. |
| `shared_pages_extra_bytes` | `compressed_bytes - physical_compressed_bytes - virtual_bytes`: bytes counted more than once. |
| page `shared_count` | number of columns that reference this page |
| page `is_owner` | `true` for the first column that references the page; count physical bytes at the owner only |
| page `shared_with` | up to 8 other fields that share the page |

A consumer that credits a reread of a shared page to every field double counts. The Darshan script
splits it equally among the sharing fields.

### Virtual pages (offset 0)

Pages reported with `offset == 0` cannot be real file positions (the file header lives there). They
are written with `"virtual": true`, counted in `virtual_pages` / `virtual_bytes`, and **excluded from
the physical total**. Their origin is not understood (an AOD has 372 of them, 51,328 bytes, stored
uncompressed). They are flagged rather than mapped.

### `field_totals`

Keyed by field name. A field with several columns gets `name#col<id>` for the additional columns, so
nothing is overwritten and the totals add up to the cluster totals.

## Fixed bugs (compared with the first version of the dumper)

| Problem | Effect | Fix |
|---|---|---|
| Early return when the next basket starts after the cluster's first entry | Baskets starting inside the **last** cluster were never counted (7,538 bytes = 6 `EventInfo` baskets missing in an EVNT file) | Assign each basket to the cluster containing its first entry |
| Array length from `GetMaxBaskets()` | Zero-filled tail of the ROOT arrays breaks `np.searchsorted`; tiny trees reported `baskets: 10` instead of 1 | Use `GetWriteBasket()` |
| Only top-level branches walked | Baskets of sub-branches missing from the volume | Recurse into `GetListOfBranches()` |
| TTree byte ranges were per-tree cursors | Overlap between collections | Added `file_ranges` from `GetBasketSeek(i)` and `GetBasketBytes()[i]` |
| `field_totals` keyed by field name | Columns of the same field overwrote each other (e.g. 457 instead of 4,783 bytes in `##Links`) | Unique key per column |
| Cluster `uncompressed_bytes` was a copy of the compressed size | Wrong value | Real uncompressed sum |
| `GetPosition<unsigned long long>` fallback | Compile error spam on ROOT 6.36 (position is `variant<uint64_t, ...>`) | `std::uint64_t`, then `unsigned long` |
| `GetFullName()` is a `TString` in recent ROOT | `json.dump` failed | `str(...)` |

## Verification

Two scripts check a dump against the ROOT file and against itself.

```
python verify_root_dump.py input.root dump.json                  # volumes, against ROOT
python verify_root_dump.py --no-root --file-size N dump.json     # internal checks only
python verify_names.py dump.json [--old old_dump.json] [--root input.root]   # branch / field names
```

`verify_root_dump.py` checks, per collection:

- TTree: dumped total == sum of `GetBasketBytes()` over all (sub-)branches == `tree.GetZipBytes()`;
  the true basket ranges do not overlap; how much of the file is not basket data.
- RNTuple: cluster, field and page sums agree; the dump equals `RNTupleInspector`; the dumper's own
  `physical_compressed_bytes`, `shared_pages_extra_bytes`, `virtual_*` agree with the pages; distinct
  page ranges do not overlap and fit in the file size.

`verify_names.py` checks that names in the clusters match the totals, flags empty or
`<unknown field N>` names and duplicates, diffs against an older dump, and compares with the names
ROOT reports.

### Results on ATLAS test files

EVNT TTree (`valid1.410000...EVNT.08166201._000012.pool.root.1`, 9 collections, file 113,750,179 B):

| Check | Result |
|---|---|
| Dump == `GetZipBytes()` for all 9 trees | yes (`CollectionTree` 112,549,611 B; `EventInfo_p4_McEventInfo` 42,169 B in 33 baskets) |
| Official `checkFile` disk sizes (kb x 1024) | match for every listed item |
| True basket ranges | 569 ranges, 112,812,080 B, no overlap across all collections |
| Not basket data | 938,099 B (0.82 %): headers, keys |
| Names | identical to the previous dump |

AOD RNTuple (`AOD.44964643._000661.pool.root.1`, file 3,819,487,109 B, `EventData` 10,000 events):

| Quantity | Bytes |
|---|---|
| `EventData` logical (== `RNTupleInspector`) | 3,897,269,261 |
| `EventData` physical (shared pages once) | 3,817,922,473 |
| Shared pages counted more than once | 79,295,460 (2.03 %); 10,051 of 98,659 distinct page ranges |
| Virtual pages (offset 0) | 372 pages, 51,328 |
| Distinct pages, all 9 collections | 3,817,993,743 |
| Not page data (header, footer, anchors) | 1,493,366 (0.04 %) |

The logical `EventData` total is larger than the file itself because of the shared pages; the
physical total plus the 0.04 % overhead accounts for the whole file.

## Use from the Darshan reread analysis

`plot_offset_histogram.py --enable_mapping --tree_branch_file dump.json`:

- TTree dumps with `file_ranges` and RNTuple dumps map at true file offsets. `--collection all` is
  then exact: no double counting, bytes outside any entry are reported as unmapped.
- Dumps without `file_ranges` fall back to the cursors with a warning naming the collections.
- With several collections and no `--collection`, `CollectionTree` is the default; if the file has no
  `CollectionTree` and every collection carries file offsets, all collections are mapped.
- Shared RNTuple pages split a reread equally among the fields that share the page; pages at offset 0
  are skipped and counted in the log.

## Known limitations

- The offset-0 RNTuple pages are unexplained (see above).
- `max_workers` is unused and `RNTupleFieldMap.__init__` / `bytesInNextCluster` are dead code.
- Verified on the EVNT TTree and AOD RNTuple above. Files with deeply split branches or other
  locator types (for example object64/DAOS) are untested; a non-file locator would be reported as
  offset 0.
- `file_ranges` makes the JSON larger: one entry per contiguous basket range, per branch and cluster.