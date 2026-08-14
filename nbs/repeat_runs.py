"""Run the same specimens repeatedly through both pipelines and aggregate.

Answers four related questions from one dataset:

  show    what ten runs of one specimen actually look like, side by side
  spread  how much each field moves between runs of the same pipeline
  vote    whether a majority vote across runs beats a single run
  differ  whether the two pipelines are independent enough to check each other

Repeated runs are not free re-reads: each one is a full pipeline call. Keep the
specimen count modest and the run count small.

    python nbs/repeat_runs.py --collect --n 30 --runs 5    # paid
    python nbs/repeat_runs.py                              # analyse, free
    python nbs/repeat_runs.py --show <occid>               # the ten-row table

Caches to transcription/results/repeat_runs/<pipeline>/<occid>/<run>.json and skips
work already done, so an interrupted collection resumes.
"""
import argparse
import json
import os
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv(Path("transcription/.env"))
import claude_sonnet as cs  # noqa: E402
from confidence import _digits, _event_year, _tokenize  # noqa: E402
from model_run import sample_occids, usable_occids  # noqa: E402
from sonnet5_hallucination import binomial, is_unknown, load_gt  # noqa: E402

OUT = "transcription/results/repeat_runs"
IMG = "transcription/data/gbif-ne-500"
ANTHROPIC_MODEL = "claude-sonnet-5"
FIELDS = ["scientificName", "eventDate", "recordedBy", "barcode", "location"]
GT_FILE = {"scientificName": "taxons.txt", "eventDate": "dates.txt",
           "recordedBy": "collectors.txt", "barcode": "catalognumbers.txt",
           "location": "localities.txt"}


def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def flat_loc(v):
    if isinstance(v, dict):
        def c(k):
            x = v.get(k)
            return "" if (x is None or str(x).strip().upper() in ("", "UNKNOWN")) else str(x).strip()
        return ", ".join(p for p in (c("locality"), c("county"), c("stateProvince")) if p)
    return "" if v is None else str(v)


def same(field, a, b):
    """Are two reads the same answer, allowing for formatting differences?"""
    if is_unknown(a) or is_unknown(b):
        return is_unknown(a) and is_unknown(b)
    if field == "scientificName":
        return ratio(binomial(a), binomial(b)) >= 0.9
    if field == "barcode":
        return _digits(a).lstrip("0") == _digits(b).lstrip("0")
    if field == "eventDate":
        return _event_year(a) == _event_year(b)
    ta = {t for t in _tokenize(a) if len(t) >= 3}
    tb = {t for t in _tokenize(b) if len(t) >= 3}
    if not ta or not tb:
        return False
    hit = sum(1 for t in ta if any(ratio(t, q) >= 0.85 for q in tb))
    return hit / min(len(ta), len(tb)) >= (0.5 if field == "recordedBy" else 0.4)


def correct_val(field, pred, gt):
    if is_unknown(pred) or is_unknown(gt):
        return None
    return same(field, pred, gt)


def run_path(pipeline, occid, i):
    return os.path.join(OUT, pipeline, occid, f"{i}.json")


def collect(occids, runs, pipelines=("anthropic", "azure")):
    import doc_intelligence as di
    total = paid = 0
    for occid in occids:
        img = os.path.join(IMG, occid + ".jpeg")
        for i in range(runs):
            for pipeline in pipelines:
                p = run_path(pipeline, occid, i)
                total += 1
                if os.path.exists(p):
                    continue
                os.makedirs(os.path.dirname(p), exist_ok=True)
                try:
                    if pipeline == "anthropic":
                        raw = cs.run_claude_pipeline(img, model=ANTHROPIC_MODEL)
                        data = cs.extract_json(raw) or {"_error": str(raw)[:300]}
                    else:
                        raw = di.run_doc_intell_pipeline(img)
                        data = cs.extract_json(raw) or {"_error": str(raw)[:300]}
                except Exception as e:
                    data = {"_error": str(e)[:300]}
                paid += 1
                if "_error" in data:
                    # Not cached: a failed call should be retried on the next run,
                    # not recorded as done.
                    print(f"  FAIL {pipeline:9} {occid} run {i}: "
                          f"{str(data['_error'])[:120]}", flush=True)
                    continue
                json.dump(data, open(p, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"  {pipeline:9} {occid} run {i}", flush=True)
    print(f"\n{paid} new calls, {total - paid} already cached")


def load_runs():
    """{pipeline: {occid: [run dicts]}}"""
    out = {}
    for pipeline in ("anthropic", "azure"):
        base = os.path.join(OUT, pipeline)
        if not os.path.isdir(base):
            continue
        per = {}
        for occid in sorted(os.listdir(base)):
            d = os.path.join(base, occid)
            if not os.path.isdir(d):
                continue
            rs = []
            for fn in sorted(os.listdir(d), key=lambda x: int(x.split(".")[0])):
                r = json.load(open(os.path.join(d, fn), encoding="utf-8"))
                if isinstance(r, dict) and "_error" not in r:
                    r = dict(r)
                    r["location"] = flat_loc(r.get("location"))
                    rs.append(r)
            if rs:
                per[occid] = rs
        out[pipeline] = per
    return out


def vote(field, values):
    """Modal value, grouping reads that mean the same thing."""
    vals = [v for v in values if not is_unknown(v)]
    if not vals:
        return ""
    groups = []
    for v in vals:
        for g in groups:
            if same(field, v, g[0]):
                g.append(v)
                break
        else:
            groups.append([v])
    return max(groups, key=len)[0]


def show(data, occid):
    print(f"\nSpecimen {occid} — every run, both pipelines\n")
    for pipeline in ("azure", "anthropic"):
        rs = data.get(pipeline, {}).get(occid, [])
        for i, r in enumerate(rs):
            head = f"{pipeline}[{i}]"
            print(f"  {head:14} " + " | ".join(
                f"{f[:12]}={str(r.get(f, ''))[:26]}" for f in FIELDS))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", default=None)
    ap.add_argument("--pipeline", nargs="+", default=["anthropic", "azure"],
                    choices=["anthropic", "azure"],
                    help="collect only these pipelines")
    args = ap.parse_args()

    if args.collect:
        occ = sample_occids(usable_occids(), args.n, seed=args.seed)
        print(f"collecting {args.runs} runs x {len(args.pipeline)} pipeline(s) "
              f"on {len(occ)} specimens: {', '.join(args.pipeline)}")
        collect(occ, args.runs, tuple(args.pipeline))
        return

    data = load_runs()
    if args.show:
        show(data, args.show)
        return

    present = [p for p in ("azure", "anthropic") if data.get(p)]
    if not present:
        raise SystemExit("no collected runs -- run with --collect first")
    occ = None
    for p in present:
        s = set(data[p])
        occ = s if occ is None else (occ & s)
    occ = sorted(occ)
    if not occ:
        raise SystemExit("no specimen has runs from every collected pipeline")
    gts = {f: load_gt(v) for f, v in GT_FILE.items()}
    nruns = min(len(data[p][o]) for p in present for o in occ)
    print(f"{len(occ)} specimens, {nruns} runs each, pipelines: {', '.join(present)}")
    if len(present) < 2:
        print("(only one pipeline collected -- the independence section needs both)")
    print()

    print("=" * 70)
    print("  SPREAD -- does a pipeline repeat itself?")
    print("=" * 70)
    print(f"  {'field':16} " + " ".join(f"{p + ' identical':>21}" for p in present))
    for f in FIELDS:
        cells = []
        for p in present:
            stable = 0
            for o in occ:
                vals = [r.get(f, "") for r in data[p][o][:nruns]]
                stable += all(same(f, vals[0], v) for v in vals[1:])
            cells.append(f"{stable / len(occ):>21.0%}")
        print(f"  {f:16} " + " ".join(cells))

    print("\n" + "=" * 70)
    print("  VOTE -- does aggregating beat one run?")
    print("=" * 70)
    print(f"  {'field':16} {'pipeline':11} {'single run':>11} {'vote of ' + str(nruns):>12} {'change':>8}")
    for f in FIELDS:
        for p in present:
            s_ok = v_ok = n = 0
            for o in occ:
                gt = gts[f].get(o)
                rs = data[p][o][:nruns]
                cs_ = correct_val(f, rs[0].get(f, ""), gt)
                cv = correct_val(f, vote(f, [r.get(f, "") for r in rs]), gt)
                if cs_ is None or cv is None:
                    continue
                n += 1
                s_ok += cs_
                v_ok += cv
            if n:
                print(f"  {f:16} {p:11} {s_ok/n:>10.0%} {v_ok/n:>12.0%} "
                      f"{(v_ok - s_ok)/n:>+8.0%}   (n={n})")

    if len(present) < 2:
        print(f"\nExample table:  python nbs/repeat_runs.py --show {occ[0]}")
        return

    print("\n" + "=" * 70)
    print("  DIFFER -- are the two pipelines independent?")
    print("=" * 70)
    print(f"  {'field':16} {'runs agree across pipelines':>29}")
    for f in FIELDS:
        hits = tot = 0
        for o in occ:
            a = vote(f, [r.get(f, "") for r in data["anthropic"][o][:nruns]])
            z = vote(f, [r.get(f, "") for r in data["azure"][o][:nruns]])
            if is_unknown(a) or is_unknown(z):
                continue
            tot += 1
            hits += same(f, a, z)
        if tot:
            print(f"  {f:16} {hits/tot:>28.0%}   (n={tot})")

    print(f"\nExample table:  python nbs/repeat_runs.py --show {occ[0]}")


if __name__ == "__main__":
    main()
