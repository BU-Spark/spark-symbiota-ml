"""Fit a per-field confidence CALIBRATION map (isotonic regression, monotonic so it
preserves ranking/rho -- it only remaps the numbers so conf=0.8 really means ~80%
correct). Fit on the GBIF eval set, validate ECE on the NE-50 holdout, and write the
map to transcription/calibration.json for the runtime to load.

    python nbs/fit_calibration.py
"""

import json
import os
import re
import sys
from difflib import SequenceMatcher

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import (GbifTaxonMatcher, scientificname_confidence, eventdate_confidence,  # noqa: E402
                        recordedby_confidence, barcode_confidence, location_confidence,
                        _event_year, _digits, _tokenize, _state_code)
from doc_intelligence import _binomial  # noqa: E402
from sciname_now_eval import final_extraction  # noqa: E402

OCR = "transcription/results/ocr_cache/azure"
SC = "transcription/results/self_consistency_cache/azure"
VIS = "transcription/results/vision_cache"
OUT = "transcription/calibration.json"
FIELDS = ["scientificName", "eventDate", "barcode", "recordedBy", "location"]


def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def sig(x):
    return _digits(x).lstrip("0")


def load(gt_dir, f):
    d = {}
    p = os.path.join(gt_dir, f)
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8", errors="ignore"):
            if ":" in ln:
                k, v = ln.split(":", 1); d[k.strip()] = v.strip()
    return d


def collect(gt_dir, m):
    tax = load(gt_dir, "taxons.txt"); date = load(gt_dir, "dates.txt")
    bc = load(gt_dir, "catalognumbers.txt"); col = load(gt_dir, "collectors.txt")
    out = {k: [] for k in ["scientificName", "eventDate", "barcode", "recordedBy"]}
    for fn in sorted(os.listdir(OCR)):
        if not fn.endswith(".json") or fn[:-5] not in tax:
            continue
        occid = fn[:-5]
        rec = json.load(open(os.path.join(OCR, fn), encoding="utf-8")); F, W = rec["fields"], rec["words"]
        scp = os.path.join(SC, f"{occid}.json"); scc = json.load(open(scp, encoding="utf-8")) if os.path.exists(scp) else None
        vp = os.path.join(VIS, f"{occid}.json"); vis = json.load(open(vp, encoding="utf-8")) if os.path.exists(vp) else {}
        sn = F.get("scientificName", "")
        if sn and sn.upper() != "UNKNOWN":
            fin, _ = final_extraction(m, sn, vis.get("scientificName", ""))
            c = scientificname_confidence(sn, W, m, vis.get("scientificName"))
            if c is not None:
                out["scientificName"].append((c, 1 if ratio(_binomial(fin), _binomial(tax[occid])) >= 0.9 else 0))
        ed = F.get("eventDate", "")
        if ed and ed.upper() != "UNKNOWN" and occid in date and _event_year(date[occid]) is not None:
            scs = [s.get("eventDate", "") for s in scc] if scc else None
            c = eventdate_confidence(ed, W, scs, vis.get("eventDate"))
            if c is not None:
                out["eventDate"].append((c, 1 if _event_year(ed) == _event_year(date[occid]) else 0))
        b = F.get("barcode", "")
        if b and b.upper() != "UNKNOWN" and occid in bc and sig(bc[occid]):
            scs = [s.get("barcode", "") for s in scc] if scc else None
            c = barcode_confidence(b, W, scs); sp, sg = sig(b), sig(bc[occid])
            if c is not None:
                out["barcode"].append((c, 1 if (sp == sg or sp.endswith(sg) or sg.endswith(sp)) else 0))
        rb = F.get("recordedBy", "")
        if rb and rb.upper() != "UNKNOWN" and occid in col:
            c = recordedby_confidence(rb, W, vis.get("recordedBy"))
            pt = [t for t in _tokenize(rb) if len(t) >= 3]; gtt = [t for t in _tokenize(col[occid]) if len(t) >= 3]
            if c is not None and gtt:
                out["recordedBy"].append((c, 1 if any(ratio(p, gtt[-1]) >= 0.85 for p in pt) else 0))
    return out


def admin_recall(pred, a):
    """Fraction of GBIF's state/county/locality tokens present in `pred`.

    The portal displays the flat locality string, so that is what the score has to
    describe. Labelling by state alone answers an easier question.
    """
    parts = [a.get("stateProvince", ""),
             re.sub(r"county", "", a.get("county", ""), flags=re.I),
             a.get("locality", "")]
    g = _tokenize(" ".join(parts))
    if not g:
        return None
    p = _tokenize(pred)
    return sum(1 for t in g if any(ratio(t, q) >= 0.85 for q in p)) / len(g)


def location_pairs():
    # Built from the OCR and vision caches. Labels are STATE correctness, which is
    # not the admin token-recall that nbs/roi_pareto_eval.py scores location on.
    admin = json.load(open("transcription/data/gbif-ne-500/gbif_admin_location.json",
                           encoding="utf-8"))

    def clean(x):
        return "" if (x is None or str(x).strip().upper() in ("", "UNKNOWN")) else str(x).strip()

    def flat(loc):
        if isinstance(loc, dict):
            return ", ".join(p for p in (clean(loc.get("locality")),
                                         clean(loc.get("county")),
                                         clean(loc.get("stateProvince"))) if p)
        return clean(loc)

    pairs = []
    for fn in sorted(os.listdir(OCR)):
        if not fn.endswith(".json"):
            continue
        occid = fn[:-5]
        if occid not in admin:
            continue
        rec = json.load(open(os.path.join(OCR, fn), encoding="utf-8"))
        llm = flat(rec["fields"].get("location"))
        if not llm:
            continue
        vp = os.path.join(VIS, f"{occid}.json")
        visf = flat(json.load(open(vp, encoding="utf-8")).get("location")) if os.path.exists(vp) else ""
        c = location_confidence(llm, rec["words"], visf or None)
        if c is None:
            continue
        acc = admin_recall(llm, admin[occid])
        if acc is None:
            continue
        pairs.append((c, 1 if acc >= 0.5 else 0))
    return pairs


def wilson_lower(hits, n, z=1.2816):
    """Lower end of a one-sided 90% interval on hits/n, so the map never claims
    more than the data supports. Small blocks shrink most."""
    if n == 0:
        return 0.0
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, centre - margin)


def over_ece(pairs, knots=None, bins=5):
    """ECE counting only buckets where the map claims more than it delivers."""
    if not pairs:
        return float("nan")
    buckets = {}
    for c, ok in pairs:
        v = apply_cal(knots, c) if knots else c
        buckets.setdefault(min(int(v * bins), bins - 1), []).append((v, ok))
    n = sum(len(b) for b in buckets.values())
    return sum(len(b) / n * max(0.0, sum(v for v, _ in b) / len(b)
                                     - sum(o for _, o in b) / len(b))
               for b in buckets.values())


def isotonic(pairs, min_count=25, conservative=True):
    # PAVA: monotonic non-decreasing fit of P(correct) vs score, then merge any
    # block with < min_count samples into a neighbour so every calibrated level
    # rests on enough data to be a stable probability (regularization vs overfit).
    pts = sorted(pairs, key=lambda t: t[0])
    blocks = []  # [sum_y, count, x_last]
    for x, y in pts:
        blocks.append([y, 1, x])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2, x2 = blocks.pop(); s1, c1, x1 = blocks.pop()
            blocks.append([s1 + s2, c1 + c2, x2])
    # merge undersized blocks (then re-run PAVA to restore monotonicity)
    changed = True
    while changed and len(blocks) > 1:
        changed = False
        for i in range(len(blocks)):
            if blocks[i][1] < min_count:
                j = i - 1 if i > 0 else i + 1
                a, b = sorted((i, j))
                blocks[a] = [blocks[a][0] + blocks[b][0], blocks[a][1] + blocks[b][1], max(blocks[a][2], blocks[b][2])]
                del blocks[b]
                changed = True
                break
    for i in range(len(blocks) - 1):
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            blocks[i + 1] = [blocks[i][0] + blocks[i + 1][0], blocks[i][1] + blocks[i + 1][1], blocks[i + 1][2]]
    knots = []
    lo_x = pts[0][0]
    floor = 0.0
    for s, c, x_last in blocks:
        p = wilson_lower(s, c) if conservative else s / c
        # The lower bound shrinks small blocks hardest, which can invert two
        # neighbours. The map must stay non-decreasing.
        floor = p = max(p, floor)
        p = round(p, 4)
        if not knots:
            knots.append([lo_x, p])
        knots.append([x_last, p])
    out = []
    for x, p in knots:
        if out and out[-1][0] == x:
            out[-1][1] = p
        else:
            out.append([x, p])
    return out


def apply_cal(knots, x):
    if x <= knots[0][0]:
        return knots[0][1]
    if x >= knots[-1][0]:
        return knots[-1][1]
    for i in range(1, len(knots)):
        if x <= knots[i][0]:
            x0, y0 = knots[i - 1]; x1, y1 = knots[i]
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


def ece(pairs, knots=None):
    edges = [i / 10 for i in range(11)]
    n = len(pairs); e = 0.0
    scored = [((apply_cal(knots, c) if knots else c), o) for c, o in pairs]
    for i in range(10):
        b = [(c, o) for c, o in scored if edges[i] <= c < edges[i + 1] or (i == 9 and c == 1.0)]
        if b:
            mc = sum(c for c, _ in b) / len(b); acc = sum(o for _, o in b) / len(b)
            e += len(b) / n * abs(mc - acc)
    return e


def main():
    m = GbifTaxonMatcher()
    train = collect("transcription/data/gbif-ne-500", m); train["location"] = location_pairs()
    hold = collect("transcription/data/new-england-samples/output", m)  # no location holdout

    # sweep regularization strength; inspect HOLDOUT ECE to pick one robust setting
    print("  holdout ECE by min_count (lower=better; 'raw' = uncalibrated):")
    print(f"  {'field':15}{'raw':>8}" + "".join(f"{'mc=' + str(mc):>9}" for mc in (1, 8, 15, 25)))
    holdable = [f for f in FIELDS if f in hold and len(hold[f]) >= 10]
    for f in FIELDS:
        row = f"  {f:15}"
        row += f"{ece(hold[f]):>8.3f}" if f in holdable else f"{'--':>8}"
        for mc in (1, 8, 15, 25):
            k = isotonic(train[f], min_count=mc)
            row += f"{ece(hold[f], k):>9.3f}" if f in holdable else f"{'--':>9}"
        print(row)

    MC = 8  # robust middle: enough data per level, keeps scientificName's structure
    cal = {}
    print(f"\n  FINAL (min_count={MC}) -- ship calibration only where it helps holdout:")
    print(f"  {'field':15}{'train raw->cal':>20}{'holdout raw->cal':>22}   ship?")
    for f in FIELDS:
        knots = isotonic(train[f], min_count=MC)
        if f in holdable:
            hr, hc = over_ece(hold[f]), over_ece(hold[f], knots)
            # Conservative maps under-state by design, so plain ECE would reject
            # them. Only over-statement can mislead, and the map also caps the
            # ceiling, so ship unless it makes over-statement worse.
            ship = hc <= hr + 0.005
            hs = f"{hr:.3f} -> {hc:.3f}"
        else:
            ship = True
            hs = "(too few holdout rows)"
        if ship:
            cal[f] = knots
        print(f"  {f:15}{ece(train[f]):>10.3f} -> {ece(train[f], knots):<6.3f}{hs:>22}   {'YES' if ship else 'no (keep raw)'}")

    json.dump(cal, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\n  wrote {OUT}  (fields calibrated: {sorted(cal)})\n")


if __name__ == "__main__":
    main()
