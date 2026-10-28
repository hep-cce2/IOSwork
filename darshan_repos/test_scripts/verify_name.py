#!/usr/bin/env python3
"""
Verify branch / field names in a ROOT cluster dump (TTree branches, RNTuple fields).

usage:
  python verify_name.py dump.json                       # internal checks only
  python verify_name.py dump.json --old old_dump.json   # + names added/removed/renamed vs an older dump
  python verify_name.py dump.json --root input.root     # + compare with the names ROOT itself reports

Checks (JSON-only part tested on the EVNT TTree and AOD RNTuple dumps; the --root part is UNTESTED,
no ROOT was available when it was written):
  1. names used in clusters[*] == names in branch_totals / field_totals  (RNTuple: totals may carry '#col<N>')
  2. empty names, '<unknown field N>' placeholders, leading/trailing spaces
  3. TTree: the same branch name twice in one cluster (would be merged by consumers)
  4. --old : names only in the old dump / only in the new one; likely renames (one name contains the other as prefix or suffix)
  5. --root: TTree  -> GetFullName() vs GetName() for every (sub-)branch, and dump names == ROOT names
             RNTuple-> dump field names == fields that own columns in the descriptor
"""
import argparse, json, re, sys


def names_in_dump(d):
    """(names used in clusters, names in totals, list of per-cluster duplicate names)."""
    used, dup = set(), []
    for c in d["clusters"]:
        lst = [b["branch"] for b in c.get("branches", [])] or None
        if lst is not None:
            used.update(lst)
            seen = set()
            for n in lst:
                if n in seen:
                    dup.append((c["cluster_index"], n))
                seen.add(n)
        else:
            used.update(f["field"] for f in c.get("fields", []))
    totals = set(d.get("branch_totals", d.get("field_totals", {})))
    return used, totals, dup


def strip_col(n):
    return re.sub(r"#col\d+$", "", n)


def ttree_root_names(f, name):
    def walk(brs):
        for b in brs:
            yield b
            yield from walk(b.GetListOfBranches())
    t = f.Get(name)
    full, differ = set(), []
    for b in walk(t.GetListOfBranches()):
        gn, gf = str(b.GetName()), str(b.GetFullName())
        full.add(gf)
        if gn != gf:
            differ.append((gn, gf))
    return full, differ


def rntuple_root_names(name, input_file):
    import ROOT
    reader = ROOT.RNTupleReader.Open(name, input_file)
    desc = reader.GetDescriptor()
    names, owners = {}, set()

    def walk(fid, prefix):
        if fid == 0xFFFFFFFF:
            return
        fd = desc.GetFieldDescriptor(fid)
        qn = f"{prefix}.{fd.GetFieldName()}" if prefix else fd.GetFieldName()
        names[int(fid)] = qn
        for ch in fd.GetLinkIds():
            walk(int(ch), qn)
    for top in desc.GetFieldZero().GetLinkIds():
        walk(int(top), "")
    for cid in desc.GetClusterIterable():
        cd = desc.GetClusterDescriptor(cid.GetId())
        for cr in cd.GetColumnRangeIterable():
            owners.add(names.get(int(desc.GetColumnDescriptor(cr.GetPhysicalColumnId()).GetFieldId())))
    return {n for n in owners if n}


def show(label, items, limit=8):
    items = sorted(items)
    print(f"    {label}: {len(items)}")
    for x in items[:limit]:
        print(f"        {x}")
    if len(items) > limit:
        print(f"        ... {len(items) - limit} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--old")
    ap.add_argument("--root")
    a = ap.parse_args()
    new = json.load(open(a.dump))
    cols = new.get("collections", {new.get("tree_name", "?"): new})
    old_cols = None
    if a.old:
        o = json.load(open(a.old))
        old_cols = o.get("collections", {o.get("tree_name", "?"): o})
    f = None
    if a.root:
        import ROOT
        f = ROOT.TFile.Open(a.root)

    problems = 0
    for n, d in cols.items():
        kind = "TTree" if d.get("type") == "TTree" else "RNTuple"
        used, totals, dup = names_in_dump(d)
        used = {str(x) for x in used}
        issues = []
        base_totals = {strip_col(t) for t in totals}
        if used != base_totals:
            issues.append(("in clusters but not in totals", used - base_totals))
            issues.append(("in totals but not in clusters", base_totals - used))
        bad = [x for x in used if not x or x != x.strip() or x.startswith("<unknown")]
        if bad:
            issues.append(("empty / padded / <unknown field> names", bad))
        if kind == "TTree" and dup:
            issues.append(("same branch name twice in one cluster", [f"cluster {c}: {x}" for c, x in dup]))
        extra = ""
        if old_cols is not None and n in old_cols:
            ou, _, _ = names_in_dump(old_cols[n])
            removed, added = ou - used, used - ou
            ren = [(r, x) for r in removed for x in added if x.endswith(r) or r.endswith(x) or x.startswith(r) or r.startswith(x)]
            if removed:
                issues.append(("names in the OLD dump missing from the new one", removed))
            extra = f"  (old {len(ou)}, new {len(used)}, added {len(added)}, removed {len(removed)}"
            extra += f", likely renames {len(ren)})" if ren else ")"
            if ren:
                issues.append(("likely renames (old -> new)", [f"{r} -> {x}" for r, x in ren]))
        if f is not None:
            if kind == "TTree":
                rn, differ = ttree_root_names(f, n)
                rn = {str(x) for x in rn}
                if differ:
                    issues.append(("GetName() != GetFullName() (examples)", [f"{x} vs {y}" for x, y in differ]))
                if rn != used:
                    issues.append(("ROOT branches missing from dump", [repr(x) for x in rn - used]))
                    issues.append(("dump branches ROOT does not report", [repr(x) for x in used - rn]))
            else:
                rn = {str(x) for x in rntuple_root_names(n, a.root)}
                if rn != used:
                    issues.append(("ROOT fields (with columns) missing from dump", rn - used))
                    issues.append(("dump fields ROOT does not report", used - rn))
        issues = [(l, i) for l, i in issues if i]
        print(f"{n:22s} {kind:8s} {len(used):7d} names  {'OK' if not issues else 'ISSUES'}{extra}")
        for label, items in issues:
            problems += 1
            show(label, items)
    print(f"\n{problems} issue group(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())