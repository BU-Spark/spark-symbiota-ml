"""Pick a stratified specimen sample out of a GBIF occurrence export.

The export is a multi-GB tab-separated dump with no image URLs; this streams it
once and writes a list of gbifIDs that nbs/gbif_refetch_occids.py resolves to
images and ground truth.

Stratified by institution: one herbarium means one label format, so a uniform
draw over the export can span very few label styles.

    # 50 New England specimens for the hand-labelling pilot
    python nbs/sample_from_export.py --csv ~/Downloads/data.csv --n 50 \
        --states Massachusetts Connecticut "Rhode Island" Vermont \
                 "New Hampshire" Maine --out nbs/pilot_occids.txt

    # 300 non-US specimens for the out-of-domain test
    python nbs/sample_from_export.py --csv ~/Downloads/data.csv --n 300 \
        --exclude-countries US --out nbs/ood_occids.txt

Free: reads a local file, no API calls.
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="GBIF export (tab-separated)")
    ap.add_argument("--out", required=True, help="file to write gbifIDs to, one per line")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--states", nargs="*", default=None, help="stateProvince filter")
    ap.add_argument("--countries", nargs="*", default=None, help="countryCode allow-list")
    ap.add_argument("--exclude-countries", nargs="*", default=None)
    ap.add_argument("--year-lo", type=int, default=None)
    ap.add_argument("--year-hi", type=int, default=None)
    ap.add_argument("--per-institution-cap", type=int, default=60,
                    help="reservoir size held per institution while streaming")
    ap.add_argument("--institutions", type=int, default=None,
                    help="draw from at most this many institutions. Default spreads "
                         "one specimen per institution, which stresses label variety "
                         "but does not match the corpus mix; set e.g. 10 to get "
                         "several specimens from each of the larger collections")
    ap.add_argument("--exclude-from", nargs="*", default=None,
                    help="directories whose occids to exclude, so the draw is "
                         "disjoint from an existing set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    states = set(args.states) if args.states else None
    allow = set(args.countries) if args.countries else None
    deny = set(args.exclude_countries) if args.exclude_countries else set()
    rng = random.Random(args.seed)

    exclude = set()
    for d in (args.exclude_from or []):
        for f in os.listdir(d):
            exclude.add(os.path.splitext(f)[0])
    if exclude:
        print(f"excluding {len(exclude)} occids already covered")

    # Bounded reservoir per institution: memory stays flat on a file larger than RAM.
    pool = defaultdict(list)
    seen = defaultdict(int)
    scanned = kept = 0

    with open(args.csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            scanned += 1
            if "StillImage" not in (row.get("mediaType") or ""):
                continue
            cc = row.get("countryCode") or ""
            if allow and cc not in allow:
                continue
            if cc in deny:
                continue
            if states and (row.get("stateProvince") or "") not in states:
                continue
            y = row.get("year") or ""
            if args.year_lo or args.year_hi:
                if not y.isdigit():
                    continue
                y = int(y)
                if args.year_lo and y < args.year_lo:
                    continue
                if args.year_hi and y > args.year_hi:
                    continue
            gid = (row.get("gbifID") or "").strip()
            if not gid or gid in exclude:
                continue

            inst = (row.get("institutionCode") or "?").strip() or "?"
            kept += 1
            seen[inst] += 1
            res = pool[inst]
            if len(res) < args.per_institution_cap:
                res.append(gid)
            else:
                j = rng.randrange(seen[inst])
                if j < args.per_institution_cap:
                    res[j] = gid

    if not kept:
        raise SystemExit("no rows matched the filters")

    # Round-robin across institutions so no single collection dominates.
    order = sorted(pool, key=lambda k: -seen[k])
    if args.institutions:
        order = order[:args.institutions]
    for k in order:
        rng.shuffle(pool[k])
    picked, i = [], 0
    while len(picked) < args.n:
        added = False
        for k in order:
            if i < len(pool[k]):
                picked.append((k, pool[k][i]))
                added = True
                if len(picked) >= args.n:
                    break
        if not added:
            break
        i += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for _, gid in picked:
            f.write(gid + "\n")

    comp = Counter(k for k, _ in picked)
    print(f"scanned {scanned:,} rows, {kept:,} matched the filters")
    print(f"wrote {len(picked)} gbifIDs to {args.out}")
    print(f"spanning {len(comp)} institutions:")
    for k, v in comp.most_common(12):
        print(f"   {k[:44]:<46}{v:>4}")
    if len(comp) > 12:
        print(f"   ... and {len(comp) - 12} more")


if __name__ == "__main__":
    main()
