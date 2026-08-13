"""Does agreement between two Anthropic models predict whether the answer is right?

The Anthropic pipeline has no confidence score. The Azure pipeline's strongest
signals are all reference checks -- compare the answer against an independent second
read -- so the same shape should work here, using a second model as the reference.
If it does, the Anthropic pipeline gets a confidence signal without needing Azure.

For each ordered pair (primary, checker) and each field: does the primary's answer
agree with the checker's, and is the primary more often correct when they agree?

  acc | agree     accuracy of the primary where the checker agrees
  acc | disagree  accuracy of the primary where the checker does not
  lift            the difference -- how much agreement tells you
  rho             Spearman of agreement against correctness, comparable to the
                  Azure per-field signals (+0.36 to +0.48)

Agreement is measured on the RAW reads. The GBIF taxon correction snaps different
synonyms onto one accepted name, which would inflate agreement on scientificName for
free. Correctness is measured on the corrected value, which is what each pipeline
ships; the two differ only for scientificName.

    python nbs/model_agreement.py
    python nbs/model_agreement.py --field scientificName --pairs-only

Free: reads cached model outputs, no API calls.
"""
import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from itertools import permutations

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import GbifTaxonMatcher, _digits, _event_year, _tokenize  # noqa: E402
from sonnet5_hallucination import (FIELDS, get_field, is_unknown, load_gt,  # noqa: E402
                                   correct, binomial, flatten_loc)
from model_run import PRICES  # noqa: E402

MODEL_CACHE = "transcription/results/model_cache"
GT_FILE = {"scientificName": "taxons.txt", "eventDate": "dates.txt",
           "recordedBy": "collectors.txt", "barcode": "catalognumbers.txt",
           "location": "localities.txt"}


def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def _token_overlap(a, b):
    # Symmetric: fraction of the shorter token set matched in the longer.
    ta = {t for t in _tokenize(a) if len(t) >= 3}
    tb = {t for t in _tokenize(b) if len(t) >= 3}
    if not ta or not tb:
        return None
    hit = sum(1 for t in ta if any(ratio(t, q) >= 0.85 for q in tb))
    return hit / min(len(ta), len(tb))


def agrees(field, a, b):
    """Do two model reads say the same thing? None when not comparable."""
    if is_unknown(a) or is_unknown(b):
        return None
    if field == "scientificName":
        return ratio(binomial(a), binomial(b)) >= 0.9
    if field == "barcode":
        da, db = _digits(a).lstrip("0"), _digits(b).lstrip("0")
        return (da == db) if (da and db) else None
    if field == "eventDate":
        ya, yb = _event_year(a), _event_year(b)
        return (ya == yb) if (ya and yb) else None
    if field in ("recordedBy", "location"):
        ov = _token_overlap(a, b)
        return None if ov is None else ov >= (0.5 if field == "recordedBy" else 0.4)
    return None


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def load_runs():
    runs = {}
    if os.path.isdir(MODEL_CACHE):
        for d in sorted(os.listdir(MODEL_CACHE)):
            p = os.path.join(MODEL_CACHE, d)
            if os.path.isdir(p):
                runs[d] = p
    return runs


def read_all(runs, matcher):
    """{model: {occid: {field: (raw_read, shipped_value)}}} over the common occids."""
    occ = None
    for d in runs.values():
        s = {f[:-5] for f in os.listdir(d) if f.endswith(".json")}
        occ = s if occ is None else (occ & s)
    occ = sorted(occ or [])
    out = {}
    for name, d in runs.items():
        per = {}
        for o in occ:
            rec = json.load(open(os.path.join(d, o + ".json"), encoding="utf-8"))
            if not isinstance(rec, dict):
                continue
            vals = {}
            for f in FIELDS:
                raw = get_field(rec, f)
                shipped = raw
                if f == "scientificName" and not is_unknown(raw):
                    fixed, _mt = matcher.correct(raw)
                    shipped = fixed or raw
                vals[f] = (raw, shipped)
            per[o] = vals
        out[name] = per
    return out, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", nargs="+", default=[f for f in FIELDS if f != "institutionCode"])
    ap.add_argument("--min-n", type=int, default=15,
                    help="skip a cell when either branch has fewer than this many rows")
    args = ap.parse_args()

    runs = load_runs()
    if len(runs) < 2:
        raise SystemExit("need at least two cached model runs")
    matcher = GbifTaxonMatcher()
    data, occ = read_all(runs, matcher)
    gts = {f: load_gt(GT_FILE[f]) for f in args.field}
    print(f"{len(runs)} models, {len(occ)} specimens common to all\n")

    for field in args.field:
        gt = gts[field]
        print(f"=== {field} ===")
        print(f"{'primary':18} {'checker':18} {'agree%':>7} {'acc|agree':>10} "
              f"{'acc|disagree':>13} {'lift':>7} {'rho':>7}")
        rows = []
        for primary, checker in permutations(runs, 2):
            ag, ok = [], []
            for o in occ:
                pa = data[primary].get(o)
                ca = data[checker].get(o)
                if not pa or not ca:
                    continue
                a = agrees(field, pa[field][0], ca[field][0])
                if a is None:
                    continue
                c = correct(field, pa[field][1], gt.get(o))
                if c is None:
                    continue
                ag.append(1 if a else 0)
                ok.append(1 if c else 0)
            n_ag = sum(ag)
            n_dis = len(ag) - n_ag
            if n_ag < args.min_n or n_dis < args.min_n:
                continue
            acc_ag = sum(o for a, o in zip(ag, ok) if a) / n_ag
            acc_dis = sum(o for a, o in zip(ag, ok) if not a) / n_dis
            r = spearman(ag, ok)
            rows.append((primary, checker, n_ag / len(ag), acc_ag, acc_dis,
                         acc_ag - acc_dis, r))
        for p, c, agr, aa, ad, lift, r in sorted(rows, key=lambda x: -x[6]):
            print(f"{p:18} {c:18} {agr:>6.0%} {aa:>10.1%} {ad:>13.1%} "
                  f"{lift:>+7.1%} {r:>+7.3f}")
        if not rows:
            print("  (no pair had enough rows on both branches)")
        print()


if __name__ == "__main__":
    main()
