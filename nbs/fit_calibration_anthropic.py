"""Fit the confidence calibration map for the Anthropic pipeline.

transcription/calibration.json was fit on the Azure pipeline's score distributions
and does not transfer: the same raw score means a different probability of being
right for a different pipeline. Without its own map, claude_sonnet ships a raw
score that a portal would render as a probability -- a scientificName of 1.0 reads
as certainty when it is about 91% correct.

Isotonic (PAVA), so the fit is monotonic: it never reorders two values, it only
remaps the numbers onto observed accuracy. Writes
transcription/calibration_anthropic.json.

    python nbs/fit_calibration_anthropic.py
    python nbs/fit_calibration_anthropic.py --cache transcription/results/model_cache/claude-sonnet-4-6

Free: reads cached model outputs.
"""
import argparse
import json
import os
import random
import re
import sys
from difflib import SequenceMatcher

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
import claude_sonnet as cs  # noqa: E402
from confidence import GbifTaxonMatcher, _tokenize  # noqa: E402
from sonnet5_hallucination import correct, get_field, is_unknown, load_gt  # noqa: E402

GT_DIR = os.environ.get("HERBARIA_GT_DIR", "transcription/data/gbif-ne-500")
OCR = "transcription/results/ocr_cache/azure"
OUT = "transcription/calibration_anthropic.json"

# scientificName and location need only the primary cache; the rest need a second
# read, so they are fit on the smaller set where both are available.
FIELDS = ["scientificName", "location", "eventDate", "recordedBy", "barcode"]
GT_FILE = {"scientificName": "taxons.txt", "eventDate": "dates.txt",
           "recordedBy": "collectors.txt", "barcode": "catalognumbers.txt"}


def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def admin_accuracy(pred, a):
    parts = [a.get("stateProvince", ""),
             re.sub(r"\bcounty\b", "", a.get("county", ""), flags=re.I),
             a.get("locality", "")]
    g = _tokenize(" ".join(parts))
    if not g:
        return None
    p = _tokenize(pred)
    return sum(1 for t in g if any(ratio(t, q) >= 0.85 for q in p)) / len(g)


def wilson_lower(hits, n, z=1.2816):
    """Lower end of a one-sided 90% interval on hits/n.

    Shipping this instead of hits/n keeps the map from claiming more than the data
    supports. Blocks with few observations shrink most, which is where
    over-confidence comes from.
    """
    if n == 0:
        return 0.0
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, centre - margin)


def isotonic(pairs, min_count=15, conservative=True):
    """PAVA fit of P(correct) vs score, then merge blocks smaller than min_count
    into a neighbour so every level rests on enough data to be a stable rate."""
    # Aggregate by score first. Sorting the raw (score, ok) tuples puts the zeros
    # before the ones inside each score group, which is already non-decreasing, so
    # PAVA would merge nothing and every observation would become its own block.
    agg = {}
    for score, ok in pairs:
        a = agg.setdefault(score, [0.0, 0.0])
        a[0] += ok
        a[1] += 1
    blocks = [[hits, n, score] for score, (hits, n) in sorted(agg.items())]
    changed = True
    while changed:                      # enforce non-decreasing
        changed = False
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
                blocks[i + 1] = [blocks[i][0] + blocks[i + 1][0],
                                 blocks[i][1] + blocks[i + 1][1], blocks[i + 1][2]]
                blocks.pop(i)
                changed = True
                i = max(i - 1, 0)
            else:
                i += 1
    i = 0                               # merge undersized blocks
    while len(blocks) > 1 and i < len(blocks):
        if blocks[i][1] < min_count:
            j = i - 1 if i > 0 else i + 1
            lo, hi = min(i, j), max(i, j)
            blocks[lo] = [blocks[lo][0] + blocks[hi][0],
                          blocks[lo][1] + blocks[hi][1], blocks[hi][2]]
            blocks.pop(hi)
            i = 0
        else:
            i += 1
    if not conservative:
        return [[round(b[2], 4), round(b[0] / b[1], 4)] for b in blocks]
    # The lower bound shrinks small blocks hardest, which can invert two
    # neighbours. The map must stay non-decreasing.
    out, floor = [], 0.0
    for b in blocks:
        floor = max(wilson_lower(b[0], b[1]), floor)
        out.append([round(b[2], 4), round(floor, 4)])
    return out


def apply_knots(knots, score):
    if not knots:
        return score
    if score <= knots[0][0]:
        return knots[0][1]
    if score >= knots[-1][0]:
        return knots[-1][1]
    for i in range(1, len(knots)):
        x1, y1 = knots[i]
        if score <= x1:
            x0, y0 = knots[i - 1]
            return y0 if x1 == x0 else y0 + (y1 - y0) * (score - x0) / (x1 - x0)
    return knots[-1][1]


def ece(pairs, knots=None, bins=5):
    """Expected calibration error: mean gap between the score and the accuracy of
    items carrying that score."""
    if not pairs:
        return float("nan")
    buckets = {}
    for s, ok in pairs:
        v = apply_knots(knots, s) if knots else s
        buckets.setdefault(min(int(v * bins), bins - 1), []).append((v, ok))
    n = sum(len(b) for b in buckets.values())
    return sum(len(b) / n * abs(sum(v for v, _ in b) / len(b)
                                - sum(o for _, o in b) / len(b))
               for b in buckets.values())


def over_ece(pairs, knots=None, bins=5):
    """ECE counting only the buckets where the map claims more than it delivers.

    Plain ECE penalises under-confidence as heavily as over-confidence, so it
    rejects a map that is conservative by design. Only over-statement can mislead
    a reviewer, so only over-statement is scored here.
    """
    if not pairs:
        return float("nan")
    buckets = {}
    for s, ok in pairs:
        v = apply_knots(knots, s) if knots else s
        buckets.setdefault(min(int(v * bins), bins - 1), []).append((v, ok))
    n = sum(len(b) for b in buckets.values())
    return sum(len(b) / n * max(0.0, sum(v for v, _ in b) / len(b)
                                     - sum(o for _, o in b) / len(b))
               for b in buckets.values())


def _read(path, occid):
    p = os.path.join(path, occid + ".json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p, encoding="utf-8"))
    return r if isinstance(r, dict) else None


def main():
    ap = argparse.ArgumentParser()
    # Defaults follow the pipeline's model roles. Swapping them fits a map for a
    # configuration that never runs.
    ap.add_argument("--cache", default=f"transcription/results/model_cache/{cs.DEFAULT_MODEL}",
                    help="primary model cache")
    ap.add_argument("--checker", default=f"transcription/results/model_cache/{cs.CHECKER_MODEL}",
                    help="checker model cache; supplies eventDate and recordedBy")
    ap.add_argument("--min-count", type=int, default=15)
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="fraction held out to decide whether a map generalises")
    ap.add_argument("--splits", type=int, default=5,
                    help="repeated holdout splits; a map ships if all but one improve")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--point-estimate", action="store_true",
                    help="ship observed accuracy instead of its lower bound")
    args = ap.parse_args()

    admin = json.load(open(os.path.join(GT_DIR, "gbif_admin_location.json"), encoding="utf-8"))
    gts = {f: load_gt(v) for f, v in GT_FILE.items()}
    matcher = GbifTaxonMatcher()
    occ = sorted(f[:-5] for f in os.listdir(args.cache) if f.endswith(".json"))
    print(f"primary {os.path.basename(args.cache)} ({len(occ)} specimens), "
          f"checker {os.path.basename(args.checker)}\n")

    pairs = {f: [] for f in FIELDS}
    for o in occ:
        rec = _read(args.cache, o)
        if rec is None:
            continue
        raw_sci = rec.get("verbatimScientificName") or get_field(rec, "scientificName")
        raw = {f: get_field(rec, f) for f in FIELDS}
        raw["scientificName"] = raw_sci
        checker = _read(args.checker, o)
        azure = _read(OCR, o)
        azure_read = azure["fields"] if azure and "fields" in azure else None
        conf = cs.build_confidence(raw, checker, azure_read, calibrate=False)

        for f in FIELDS:
            c = conf.get(f)
            if c is None or is_unknown(raw.get(f)):
                continue
            if f == "scientificName":
                shipped, _ = matcher.correct(raw_sci)
                ok = correct(f, shipped or raw_sci, gts[f].get(o))
            elif f == "location":
                acc = admin_accuracy(raw[f], admin[o]) if o in admin else None
                ok = None if acc is None else acc >= 0.5
            else:
                ok = correct(f, raw[f], gts[f].get(o))
            if ok is not None:
                pairs[f].append((c, 1 if ok else 0))

    out = {}
    print(f"  {'field':16} {'n':>5} {'holdout raw':>12} {'holdout cal':>12} "
          f"{'splits won':>11}  ship?")
    for f in FIELDS:
        p = pairs[f]
        if len(p) < 30:
            print(f"  {f:16} {len(p):>5}   too few rows to fit")
            continue
        # Fit on a subset and score on the rest, repeatedly. Isotonic always fits
        # its own training data well, so in-sample ECE says nothing about whether a
        # map generalises. One split is not enough either -- a field can improve on
        # a lucky draw and get worse on the next -- so require most splits to agree.
        wins, raws, cals, bad = 0, [], [], None
        for s in range(args.splits):
            rng = random.Random(args.seed + s)
            idx = list(range(len(p)))
            rng.shuffle(idx)
            cut = int(len(p) * (1 - args.holdout))
            train = [p[i] for i in idx[:cut]]
            test = [p[i] for i in idx[cut:]]
            knots = isotonic(train, args.min_count, not args.point_estimate)
            if len({y for _, y in knots}) < 2 or len(test) < 20:
                bad = "fit collapsed" if len({y for _, y in knots}) < 2 else "holdout too small"
                break
            score = ece if args.point_estimate else over_ece
            h0, h1 = score(test), score(test, knots)
            raws.append(h0)
            cals.append(h1)
            # Conservative mode also wants the ceiling the map imposes, so ship
            # unless the map makes over-statement worse.
            wins += (h1 <= h0 + 0.005) if not args.point_estimate else (h1 < h0)
        if bad:
            print(f"  {f:16} {len(p):>5} {'--':>12} {'--':>12} {'--':>11}  no ({bad})")
            continue
        keep = wins >= args.splits - 1          # allow one unlucky split
        full = isotonic(p, args.min_count, not args.point_estimate)
        print(f"  {f:16} {len(p):>5} {sum(raws)/len(raws):>12.3f} "
              f"{sum(cals)/len(cals):>12.3f} {wins:>6}/{args.splits}"
              f"    {'YES' if keep else 'no (keep raw)'}")
        if keep:
            out[f] = full

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}  (calibrated: {sorted(out) or 'none'})")
    for f, k in out.items():
        print(f"  {f}: " + "  ".join(f"{x:g}->{y:.2f}" for x, y in k))


if __name__ == "__main__":
    main()
