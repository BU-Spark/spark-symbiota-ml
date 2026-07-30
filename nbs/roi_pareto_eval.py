"""B3 (reviewer-hours-saved ROI) + B4 (base-vs-enhanced per-field cost/value Pareto).

Both run entirely off the cached OCR / self-consistency / vision results -- ZERO API
cost. Reuses the exact load + lenient-correctness scoring of
nbs/confidence_threshold_table.py so the numbers are consistent with the shipped
confidence.

B3: if reviewers auto-accept fields whose confidence >= a cutoff and only hand-check
    the rest, how many reviewer-hours does that save on a corpus of N specimens, and
    how many wrong values slip through unreviewed?

B4: the enhanced signals (self-consistency + vision) cost ~2x the base pipeline
    ($4.5 vs $2 / 1,000). Field by field, what does that 2x actually buy in
    discrimination (Spearman rho) and missed-error rate? -> where to spend it.

    python nbs/roi_pareto_eval.py --gt-dir transcription/data/gbif-ne-500
"""

import argparse
import json
import os
import sys
from difflib import SequenceMatcher

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import (build_confidence, location_confidence, GbifTaxonMatcher,  # noqa: E402
                        _tokenize, _digits, _event_year)
import re  # noqa: E402

OCR_CACHE = "transcription/results/ocr_cache/azure"
SC_CACHE = "transcription/results/self_consistency_cache/azure"
VIS_CACHE = "transcription/results/vision_cache"

FIELDS = ["scientificName", "eventDate", "recordedBy", "barcode"]
GT_FILE = {"scientificName": "taxons.txt", "eventDate": "dates.txt",
           "recordedBy": "collectors.txt", "barcode": "catalognumbers.txt"}


def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def load_gt(gt_dir, fn):
    d = {}
    for line in open(os.path.join(gt_dir, fn), encoding="utf-8", errors="ignore"):
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def binomial(s):
    return " ".join([t for t in _tokenize(s) if len(t) >= 3][:2])


def correct(field, pred, gt):
    # Lenient: count only real errors, not formatting/order artifacts.
    if field == "scientificName":
        return ratio(binomial(pred), binomial(gt)) >= 0.9
    if field == "barcode":
        return _digits(pred).lstrip("0") == _digits(gt).lstrip("0")
    if field == "eventDate":
        return _event_year(pred) == _event_year(gt)
    if field == "recordedBy":
        p = set(t for t in _tokenize(pred) if len(t) >= 3)
        g = [t for t in _tokenize(gt) if len(t) >= 3]
        if not g:
            return None
        return sum(any(ratio(t, q) >= 0.85 for q in p) for t in g) / len(g) >= 0.5
    return None


def spearman(xs, ys):
    # rank-correlation, ties -> average rank (no scipy).
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def load(root, occid):
    p = os.path.join(root, f"{occid}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def admin_accuracy(pred, admin):
    # Fraction of admin-GT tokens (state + county sans 'county' + locality) present
    # (fuzzy >=0.85) in the predicted location. Same metric as nbs/location_eval.py.
    parts = [admin.get("stateProvince", ""),
             re.sub(r"\bcounty\b", "", admin.get("county", ""), flags=re.I),
             admin.get("locality", "")]
    gt = _tokenize(" ".join(parts))
    if not gt:
        return None
    pt = set(_tokenize(pred))
    return sum(1 for g in gt if any(ratio(g, p) >= 0.85 for p in pt)) / len(gt)


def collect_location(gt_dir):
    """location_confidence (enhanced, vision-on) vs admin token-recall accuracy."""
    ap = os.path.join(gt_dir, "gbif_admin_location.json")
    if not os.path.exists(ap):
        return []
    admin = json.load(open(ap, encoding="utf-8"))
    out = []
    for fn in sorted(os.listdir(OCR_CACHE)):
        if not fn.endswith(".json"):
            continue
        occid = fn[:-5]
        if occid not in admin:
            continue
        rec = load(OCR_CACHE, occid)
        pred = rec["fields"].get("location", "")
        if not pred or str(pred).strip().upper() == "UNKNOWN":
            continue
        vis = load(VIS_CACHE, occid) or {}
        c = location_confidence(pred, rec["words"], vis.get("location"))
        acc = admin_accuracy(pred, admin[occid])
        if c is not None and acc is not None:
            out.append((c, acc))
    return out


def collect(gt_dir):
    """Return per-field lists of (conf_enhanced, conf_base, correct) over the set."""
    matcher = GbifTaxonMatcher()
    gts = {f: load_gt(gt_dir, GT_FILE[f]) for f in FIELDS}
    rows = {f: [] for f in FIELDS}
    n_specimens = 0
    for fn in sorted(os.listdir(OCR_CACHE)):
        if not fn.endswith(".json"):
            continue
        occid = fn[:-5]
        rec = load(OCR_CACHE, occid)
        if rec is None:
            continue
        n_specimens += 1
        sc = load(SC_CACHE, occid)
        vis = load(VIS_CACHE, occid)
        conf_enh = build_confidence(rec["fields"], rec["words"], sc_samples=sc,
                                    vision_fields=vis, taxon_matcher=matcher)
        conf_base = build_confidence(rec["fields"], rec["words"], sc_samples=None,
                                     vision_fields=None, taxon_matcher=matcher)
        for f in FIELDS:
            pred, gt = rec["fields"].get(f, ""), gts[f].get(occid)
            ce, cb = conf_enh.get(f), conf_base.get(f)
            if not gt or not pred or str(pred).upper() == "UNKNOWN":
                continue
            ok = correct(f, pred, gt)
            if ok is None:
                continue
            rows[f].append((ce, cb, 1.0 if ok else 0.0))
    return rows, n_specimens


# ---------------------------------------------------------------- B3: reviewer ROI
def b3_roi(rows, corpora, sec_per_field):
    CUT = 0.9  # operating point
    print("\n" + "=" * 72)
    print("  B3 - REVIEWER-HOURS-SAVED ROI")
    print("=" * 72)
    print(f"  Model: reviewer auto-accepts a field when confidence >= {CUT:.2f} and")
    print(f"  hand-checks the rest. Verify time = {sec_per_field}s per field.\n")

    # per-field operating characteristics at the cutoff (enhanced confidence)
    print(f"  Per-field at cutoff {CUT:.2f} (enhanced confidence):")
    print(f"    {'field':16} {'n':>5} {'auto-accept':>12} {'missed-err':>11}")
    agg_cov = 0.0
    field_stats = {}
    for f in FIELDS:
        pairs = [(ce, ok) for ce, cb, ok in rows[f] if ce is not None]
        n = len(pairs)
        kept = [ok for ce, ok in pairs if ce >= CUT]
        cov = len(kept) / n if n else 0.0
        err = 1 - sum(kept) / len(kept) if kept else 0.0
        field_stats[f] = (cov, err)
        agg_cov += cov
        print(f"    {f:16} {n:5} {cov:11.0%} {err:11.1%}")
    mean_cov = agg_cov / len(FIELDS)
    print(f"\n  Mean auto-accept coverage across {len(FIELDS)} fields: {mean_cov:.0%}")

    for N in corpora:
        # manual baseline: review every field of every specimen
        total_checks = N * len(FIELDS)
        manual_hrs = total_checks * sec_per_field / 3600.0
        # with routing: only sub-cutoff fields reviewed; skipped = coverage
        skipped = sum(field_stats[f][0] for f in FIELDS) / len(FIELDS) * total_checks
        saved_hrs = skipped * sec_per_field / 3600.0
        missed = sum(field_stats[f][0] * field_stats[f][1] for f in FIELDS) * N
        print(f"\n  --- Corpus N = {N:,} specimens ---")
        print(f"    Manual review (all fields):     {manual_hrs:10,.0f} reviewer-hours")
        print(f"    Auto-accept >= {CUT:.2f}:            saves {saved_hrs:8,.0f} hours "
              f"({saved_hrs / manual_hrs:.0%} of the work)")
        print(f"    Errors let through unreviewed:  {missed:10,.0f} field-values "
              f"({missed / total_checks:.2%} of all fields)")
        # full-time-equivalent framing (1 FTE ~ 1,760 productive hrs/yr)
        print(f"    ~= {saved_hrs / 1760:,.1f} person-years of review avoided")


# ------------------------------------------------ B4: base-vs-enhanced Pareto
def b4_pareto(rows):
    CUT = 0.9
    ENH_COST, BASE_COST = 4.5, 2.0  # $ / 1,000 specimens
    print("\n" + "=" * 72)
    print("  B4 - PER-FIELD COST/VALUE PARETO  (enhanced $4.5/1k vs base $2/1k)")
    print("=" * 72)
    print(f"  Does the extra ${ENH_COST - BASE_COST:.1f}/1k (self-consistency + vision) "
          f"improve each field?\n")
    print(f"    {'field':16} {'rho_base':>9} {'rho_enh':>9} {'d_rho':>7}   "
          f"{'miss_base':>9} {'miss_enh':>9}   verdict")
    for f in FIELDS:
        both = [(ce, cb, ok) for ce, cb, ok in rows[f] if ce is not None and cb is not None]
        if len(both) < 5:
            print(f"    {f:16}  (insufficient data)")
            continue
        rho_e = spearman([ce for ce, cb, ok in both], [ok for ce, cb, ok in both])
        rho_b = spearman([cb for ce, cb, ok in both], [ok for ce, cb, ok in both])
        ke = [ok for ce, cb, ok in both if ce >= CUT]
        kb = [ok for ce, cb, ok in both if cb >= CUT]
        me = (1 - sum(ke) / len(ke)) if ke else float("nan")
        mb = (1 - sum(kb) / len(kb)) if kb else float("nan")
        d = rho_e - rho_b
        verdict = ("ENHANCED worth it" if d >= 0.05
                   else "marginal" if d >= 0.02
                   else "BASE is enough")
        print(f"    {f:16} {rho_b:+9.3f} {rho_e:+9.3f} {d:+7.3f}   "
              f"{mb:9.1%} {me:9.1%}   {verdict}")
    print("\n  rho = discrimination (does confidence rank-predict correctness);")
    print("  miss = fraction of auto-accepted (>=0.90) values that are actually wrong.")
    print("  Note: one shared vision call + one SC batch serve ALL fields, so enhanced")
    print("  is bought per-specimen, not per-field -- use this to decide if the PACKAGE")
    print("  earns its 2x, and which fields are carrying it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="transcription/data/gbif-ne-500")
    ap.add_argument("--sec-per-field", type=float, default=10.0,
                    help="seconds a reviewer spends verifying one field")
    args = ap.parse_args()
    rows, n_spec = collect(args.gt_dir)
    print(f"\n  Loaded {n_spec} specimens from cache (zero API cost).")
    b3_roi(rows, corpora=[168_000, 1_500_000], sec_per_field=args.sec_per_field)
    b4_pareto(rows)

    # ---- ALL-5-FIELD rho summary (best metric, enhanced on) ----
    loc = collect_location(args.gt_dir)
    summary = []
    for f in FIELDS:
        both = [(ce, ok) for ce, cb, ok in rows[f] if ce is not None]
        summary.append((f, spearman([c for c, _ in both], [o for _, o in both]), len(both)))
    if loc:
        summary.append(("location", spearman([c for c, _ in loc], [a for _, a in loc]), len(loc)))
    summary.sort(key=lambda t: t[1], reverse=True)
    print("\n" + "=" * 72)
    print("  ALL 5 FIELDS - Spearman rho (best metric, ENHANCED on), best-first")
    print("=" * 72)
    for f, r, n in summary:
        print(f"    {f:16} rho = {r:+.3f}   (n={n})")
    print("\n  4 fields vs binary correctness; location vs admin token-recall")
    print("  (continuous) -- the established per-field metric convention.\n")
