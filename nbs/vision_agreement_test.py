"""Build the vision cache, and measure how well vision agreement predicts accuracy.

For N specimens: read the 6 fields straight from the image with a multimodal model
(independent of the OCR text), then per field compute agreement = similarity(vision
value, text-pipeline value) and its Spearman rho against accuracy.

Vision reads are cached to transcription/results/vision_cache/<occid>.json. That is
the cache the confidence scoring and the evals read, so this doubles as the cache
builder for the vision signal.

    python nbs/vision_agreement_test.py --n 500

Uses gpt-4o-mini (vision), a few cents per 100 images. Occids already cached are
skipped, so a re-run only fills gaps.
"""

import argparse
import base64
import io
import json
import os
import sys
from difflib import SequenceMatcher

import pandas as pd
from PIL import Image

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import _tokenize, _digits, _event_year  # noqa: E402
from location_eval import admin_accuracy  # noqa: E402

OCR_CACHE = "transcription/results/ocr_cache/azure"
VIS_CACHE = "transcription/results/vision_cache"
IMG_DIR = "transcription/data/gbif-ne-500"
FIELDS = ["recordedBy", "location", "scientificName", "eventDate", "barcode", "institutionCode"]

PROMPT = (
    "You are reading a herbarium specimen sheet. From the image, extract exactly "
    "these six fields and return ONLY a JSON object (no markdown): recordedBy "
    "(collector), location, scientificName (genus species), eventDate (ISO), "
    "barcode (catalog number), institutionCode. Use 'UNKNOWN' if a field is not "
    "legible. Format: {\"recordedBy\":...,\"location\":...,\"scientificName\":...,"
    "\"eventDate\":...,\"barcode\":...,\"institutionCode\":...}"
)


def vision_read(client, image_path):
    im = Image.open(image_path).convert("RGB")
    im.thumbnail((1536, 1536), Image.LANCZOS)  # cap tokens; labels stay legible
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
        ]}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def load_gt(fname):
    gt = {}
    with open(os.path.join(IMG_DIR, fname), encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                gt[k.strip()] = v.strip()
    return gt


def binomial(s):
    return " ".join([t for t in _tokenize(s) if len(t) >= 3][:2])


def field_accuracy(field, pred, gt):
    # Best per-field accuracy metric from the final experiment.
    if field == "scientificName":
        return 1.0 if ratio(binomial(pred), binomial(gt)) >= 0.9 else 0.0
    if field == "barcode":
        return 1.0 if _digits(pred) and _digits(pred) == _digits(gt) else 0.0
    return ratio(pred, gt)


def run(n):
    import openai
    from dotenv import load_dotenv
    load_dotenv(os.path.join("transcription", ".env"))
    client = openai.OpenAI()
    os.makedirs(VIS_CACHE, exist_ok=True)

    gts = {"scientificName": load_gt("taxons.txt"), "location": None,
           "eventDate": load_gt("dates.txt"), "recordedBy": load_gt("collectors.txt"),
           "barcode": load_gt("catalognumbers.txt")}
    admin_gt = json.load(open(os.path.join(IMG_DIR, "gbif_admin_location.json"), encoding="utf-8"))

    occids = [f[:-5] for f in sorted(os.listdir(OCR_CACHE)) if f.endswith(".json")
              and f[:-5] in gts["scientificName"]][:n]

    rows = []
    for i, occid in enumerate(occids):
        img = os.path.join(IMG_DIR, f"{occid}.jpeg")
        if not os.path.exists(img):
            continue
        vpath = os.path.join(VIS_CACHE, f"{occid}.json")
        if os.path.exists(vpath):
            vis = json.load(open(vpath, encoding="utf-8"))
        else:
            try:
                vis = vision_read(client, img)
            except Exception as e:
                print(f"  FAIL {occid}: {str(e)[:80]}")
                continue
            json.dump(vis, open(vpath, "w", encoding="utf-8"))
        text = json.load(open(os.path.join(OCR_CACHE, f"{occid}.json"), encoding="utf-8"))["fields"]

        row = {"occid": occid}
        for field in ["scientificName", "eventDate", "recordedBy", "barcode", "location"]:
            tv, vv = str(text.get(field, "")), str(vis.get(field, ""))
            if not tv or tv.upper() == "UNKNOWN":
                continue
            row[f"agree_{field}"] = ratio(tv, vv)  # cross-modal agreement signal
            if field == "location":
                if occid in admin_gt:
                    row[f"acc_{field}"] = admin_accuracy(tv, admin_gt[occid])
            elif occid in gts[field]:
                row[f"acc_{field}"] = field_accuracy(field, tv, gts[field][occid])
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(occids)}")

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} specimens with a vision read\n")
    print(f"{'field':16} {'vision-agree rho':>18} {'prev best (text-only)':>22}")
    prev = {"scientificName": 0.44, "eventDate": 0.38, "recordedBy": 0.35,
            "barcode": 0.45, "location": 0.49}
    for field in ["scientificName", "eventDate", "recordedBy", "barcode", "location"]:
        a, c = f"agree_{field}", f"acc_{field}"
        sub = df[[a, c]].dropna() if a in df and c in df else pd.DataFrame()
        if len(sub) < 5 or sub[a].nunique() < 2:
            print(f"{field:16} {'(too few)':>18} {prev[field]:>+22.2f}")
            continue
        r = sub[a].rank().corr(sub[c].rank())
        print(f"{field:16} {r:>+13.3f} (n={len(sub)}) {prev[field]:>+15.2f}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    run(args.n)
