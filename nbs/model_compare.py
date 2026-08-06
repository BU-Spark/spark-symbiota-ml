"""Score any number of cached model runs on the same specimens.

Imports the scoring from nbs/sonnet5_hallucination.py (which compares two runs) so
the numbers stay consistent: lenient per-field accuracy plus the three
hallucination measures (invented species, impossible year, ungrounded value).

Runs are scored on the intersection of the occids they cover, so a model with
fewer cached outputs cannot look better by having skipped specimens.

    python nbs/model_compare.py
    python nbs/model_compare.py --runs sonnet5=transcription/results/sonnet5_cache \
        opus5=transcription/results/model_cache/claude-opus-5

With no --runs, every directory under transcription/results/model_cache is picked
up, plus the Sonnet 5 cache and the shipped gpt-4o-mini baseline.

Free: reads cached outputs only, no API calls.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import GbifTaxonMatcher  # noqa: E402
from sonnet5_hallucination import FIELDS, OCR, S5, load_gt, score  # noqa: E402

MODEL_CACHE = "transcription/results/model_cache"


def discover_runs():
    runs = {}
    if os.path.isdir(S5):
        runs["claude-sonnet-5"] = S5
    if os.path.isdir(MODEL_CACHE):
        for d in sorted(os.listdir(MODEL_CACHE)):
            p = os.path.join(MODEL_CACHE, d)
            if os.path.isdir(p) and d not in runs:
                runs[d] = p
    return runs


def occids_in(d):
    return {f[:-5] for f in os.listdir(d) if f.endswith(".json")}


def loader(d):
    def get(occid):
        return json.load(open(os.path.join(d, occid + ".json"), encoding="utf-8"))
    return get


def mini(occid):
    return json.load(open(os.path.join(OCR, occid + ".json"), encoding="utf-8"))["fields"]


def run_cost(d, occids):
    """($ per 1,000 specimens, n measured) from the token usage cached per record."""
    from model_run import PRICES
    tin = tout = n = 0
    model = None
    for o in occids:
        p = os.path.join(d, o + ".json")
        if not os.path.exists(p):
            continue
        rec = json.load(open(p, encoding="utf-8"))
        u = rec.get("_usage") if isinstance(rec, dict) else None
        if not u:
            continue
        tin += u["input_tokens"]
        tout += u["output_tokens"]
        model = model or u.get("model")
        n += 1
    if not n:
        return None, 0
    key = next((k for k in PRICES if model and model.startswith(k)), None)
    if not key:
        return None, n
    pin, pout = PRICES[key]
    return (tin / 1e6 * pin + tout / 1e6 * pout) / n * 1000, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, metavar="NAME=DIR")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the shipped gpt-4o-mini (Azure OCR + LLM) baseline")
    args = ap.parse_args()

    if args.runs:
        runs = dict(r.split("=", 1) for r in args.runs)
    else:
        runs = discover_runs()
    runs = {k: v for k, v in runs.items() if os.path.isdir(v)}
    if not runs:
        raise SystemExit("no run directories found -- run nbs/model_run.py first")

    common = None
    for d in runs.values():
        o = occids_in(d)
        common = o if common is None else (common & o)
    common &= occids_in(OCR)          # scoring needs the OCR words for grounding
    occids = sorted(common)
    if not occids:
        raise SystemExit("no occids common to all runs")

    print(f"runs: {', '.join(runs)}")
    for name, d in runs.items():
        print(f"  {name:24} {len(occids_in(d)):4} cached")
    print(f"scored on {len(occids)} specimens common to all runs")

    gts = {"scientificName": load_gt("taxons.txt"), "eventDate": load_gt("dates.txt"),
           "recordedBy": load_gt("collectors.txt"), "barcode": load_gt("catalognumbers.txt"),
           "location": load_gt("localities.txt")}
    matcher = GbifTaxonMatcher()

    results = {}
    for name, d in runs.items():
        results[name] = score(name, loader(d), occids, gts, matcher)
    if not args.no_baseline:
        results["gpt-4o-mini (shipped)"] = score(
            "gpt-4o-mini (shipped, Azure OCR + LLM)", mini, occids, gts, matcher)

    # side-by-side accuracy summary
    names = list(results)
    w = max(len(n) for n in names) + 2
    print(f"\n\n===== accuracy summary (n={len(occids)}) =====")
    print(f"{'model':{w}}" + "".join(f"{f[:13]:>14}" for f in FIELDS) + f"{'mean':>9}")
    for name in names:
        row, accs = "", []
        for f in FIELDS:
            s = results[name][f]
            a = s["correct"] / s["scored"] if s["scored"] else float("nan")
            accs.append(a)
            row += f"{a:>13.1%} "
        mean = sum(accs) / len(accs)
        print(f"{name:{w}}{row}{mean:>8.1%}")

    print(f"\n===== cost per 1,000 specimens =====")
    for name, d in runs.items():
        cpk, n = run_cost(d, occids)
        if cpk is None:
            print(f"  {name:24} no cost recorded "
                  f"({'no token usage in cache' if not n else 'no price listed for model'})")
        else:
            print(f"  {name:24} ${cpk:7.2f}   (measured on {n} specimens)")


if __name__ == "__main__":
    main()
