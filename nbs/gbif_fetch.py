"""Fetch a labeled herbarium-specimen dataset from GBIF for confidence eval.

GBIF ``occurrence/search`` is self-labeling: one record carries both the
specimen image URL *and* the six Darwin Core ground-truth fields, from the same
herbaria as the existing new-england-samples set, so it can build an eval set of
a few hundred specimens without hand-labeling.

Focused domain (matches the current corpus): vascular plants, herbarium sheets
(PRESERVED_SPECIMEN, not iNaturalist field photos), New England US states, older
handwritten-era labels (default 1850-1950).

Writes, into ``--out-dir`` (mirrors data/new-england-samples/output layout):
  <occid>.jpeg               downloaded specimen image
  taxons.txt                 occid: scientificName    (species binomial)
  collectors.txt             occid: recordedBy
  dates.txt                  occid: eventDate
  localities.txt             occid: locality          (micro-locality)
  catalognumbers.txt         occid: catalogNumber
  institutions.txt           occid: institutionCode
  countries.txt              occid: country
  gbif_admin_location.json   occid -> {stateProvince, county, locality}

Then build the OCR cache and run the eval against this dir:
  python nbs/ocr_cache.py --pipeline azure --image-dir transcription/data/gbif-ne-500
  python nbs/confidence_eval.py --cache-dir transcription/results/ocr_cache/azure \
      --gt-dir transcription/data/gbif-ne-500

No API keys or cost here -- GBIF is a free public API. Idempotent: images and
occids already present are skipped, so a re-run only fills gaps.
"""

import argparse
import io
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# Full-res GBIF scans run 4-40 MB; downscale so images stay well under Azure's
# free-tier 4 MB limit and don't balloon disk. Label text stays legible at this
# size (the existing new-england-samples set is 0.2-4 MB).
MAX_SIDE = 2500  # px, longest edge
JPEG_QUALITY = 85

# Substrings that mark a "locality" as a GBIF placeholder, not a real place.
LOCALITY_JUNK_SUBSTR = ("no additional data", "no data", "not recorded",
                        "not captured", "not provided", "unknown", "n/a",
                        "data not captured", "s.n.")


def _is_junk_locality(v: str) -> bool:
    s = v.strip().lower()
    if len(s) <= 2:  # "-", "?", ""
        return True
    return any(j in s for j in LOCALITY_JUNK_SUBSTR)

API = "https://api.gbif.org/v1/occurrence/search"
VASCULAR_PLANTS_TAXON_KEY = 7707728  # Tracheophyta
NEW_ENGLAND = ["Connecticut", "Massachusetts", "Rhode Island",
               "Vermont", "New Hampshire", "Maine"]
USER_AGENT = "spark-symbiota-ml/confidence-eval (research; contact via BU Spark)"

# Fields we need present for a record to be useful as ground truth. locality is
# handled separately (locality OR stateProvince is acceptable).
REQUIRED = ["recordedBy", "eventDate", "institutionCode", "catalogNumber"]


def _species(rec: dict) -> str:
    # Prefer the clean canonical binomial; fall back to genus+epithet, then the
    # full scientificName (which carries the author) with author trimmed off.
    if rec.get("species"):
        return rec["species"]
    genus, epithet = rec.get("genus"), rec.get("specificEpithet")
    if genus and epithet:
        return f"{genus} {epithet}"
    return rec.get("scientificName", "")


def _event_date(rec: dict) -> str:
    # GBIF eventDate can be a range "start/end"; keep the start.
    return (rec.get("eventDate") or "").split("/")[0].strip()


def _locality(rec: dict) -> str:
    # Prefer locality, then verbatimLocality; drop GBIF junk placeholders.
    for key in ("locality", "verbatimLocality"):
        v = (rec.get(key) or "").strip()
        if v and not _is_junk_locality(v):
            return v
    return ""


def _image_url(rec: dict) -> str:
    for m in rec.get("media", []) or []:
        if m.get("type") == "StillImage" and m.get("identifier"):
            return m["identifier"]
    return ""


def _qualifies(rec: dict) -> bool:
    if not _image_url(rec):
        return False
    if not _species(rec):
        return False
    if not (_locality(rec) or rec.get("stateProvince")):
        return False
    return all(rec.get(k) for k in REQUIRED)


def _fetch_bucket(n: int, states: list, year_lo: int, year_hi: int,
                  seen: set) -> list:
    # Page through occurrence/search for one year window, keeping records with
    # full-enough GT until we have n of them (or the window is exhausted).
    kept, offset, page = [], 0, 300  # 300 = GBIF max page size
    base = {
        "taxon_key": VASCULAR_PLANTS_TAXON_KEY,
        "basis_of_record": "PRESERVED_SPECIMEN",
        "media_type": "StillImage",
        "country": "US",
        "year": f"{year_lo},{year_hi}",
        "limit": page,
    }
    while len(kept) < n:
        params = list(base.items()) + [("stateProvince", s) for s in states]
        params.append(("offset", offset))
        url = API + "?" + urllib.parse.urlencode(params, doseq=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        results = data.get("results", [])
        if not results:
            break
        for rec in results:
            occid = str(rec.get("key"))
            if occid in seen:
                continue
            if _qualifies(rec):
                seen.add(occid)
                kept.append(rec)
                if len(kept) >= n:
                    break
        offset += page
        if data.get("endOfRecords") or offset >= 100000:
            break
        time.sleep(0.2)  # be polite to the API
    return kept


def fetch_records(n: int, states: list, year_lo: int, year_hi: int,
                  exclude: set, buckets: int = 5) -> list:
    # GBIF returns newest-in-range first, so a single paged query clusters on one
    # date. Split the span into equal year windows and pull a share from each so
    # the set spreads across the handwritten era instead of piling on one year.
    seen = set(exclude)
    edges = [year_lo + round(i * (year_hi - year_lo) / buckets)
             for i in range(buckets + 1)]
    per = -(-n // buckets)  # ceil, so we slightly over-fetch then trim
    kept = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        got = _fetch_bucket(per, states, lo, hi, seen)
        kept.extend(got)
        print(f"  {lo}-{hi}: +{len(got)}  (total {len(kept)})")
    return kept[:n]


def _download(occid: str, url: str, out_dir: str) -> tuple:
    path = os.path.join(out_dir, f"{occid}.jpeg")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return occid, True, "cached"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=90) as r:
            body = r.read()
        if len(body) < 1024:  # too small to be a real scan
            return occid, False, f"tiny ({len(body)}b)"
        # Downscale + re-encode so the saved JPEG stays small enough for Azure.
        im = Image.open(io.BytesIO(body))
        im = im.convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        im.save(path, "JPEG", quality=JPEG_QUALITY)
        return occid, True, f"{os.path.getsize(path)//1024}kb {im.size}"
    except Exception as e:
        return occid, False, str(e)[:80]


def download_images(records: list, out_dir: str, workers: int = 8) -> set:
    ok = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download, str(r["key"]), _image_url(r), out_dir): r["key"]
                for r in records}
        for i, fut in enumerate(as_completed(futs), 1):
            occid, good, msg = fut.result()
            if good:
                ok.add(str(occid))
            else:
                print(f"  IMG FAIL {occid}: {msg}")
            if i % 50 == 0:
                print(f"  downloaded {i}/{len(records)}")
    return ok


def write_gt(records: list, ok: set, out_dir: str):
    # Only write GT for records whose image actually downloaded.
    recs = [r for r in records if str(r["key"]) in ok]
    recs.sort(key=lambda r: str(r["key"]))

    def dump(fname, valfn):
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            for r in recs:
                v = valfn(r)
                if v:
                    f.write(f"{r['key']}: {v}\n")

    dump("taxons.txt", _species)
    dump("collectors.txt", lambda r: r.get("recordedBy", ""))
    dump("dates.txt", _event_date)
    dump("localities.txt", _locality)
    dump("catalognumbers.txt", lambda r: r.get("catalogNumber", ""))
    dump("institutions.txt", lambda r: r.get("institutionCode", ""))
    dump("countries.txt", lambda r: r.get("country", ""))

    admin = {str(r["key"]): {
        "stateProvince": r.get("stateProvince", ""),
        "county": r.get("county", ""),
        "locality": _locality(r),
    } for r in recs}
    with open(os.path.join(out_dir, "gbif_admin_location.json"), "w",
              encoding="utf-8") as f:
        json.dump(admin, f, indent=1)
    return recs


def load_existing_occids(*gt_files) -> set:
    occ = set()
    for path in gt_files:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if ":" in line:
                        occ.add(line.split(":", 1)[0].strip())
    return occ


def summarize(recs: list):
    from collections import Counter
    inst = Counter(r.get("institutionCode", "?") for r in recs)
    state = Counter(r.get("stateProvince", "?") for r in recs)
    print(f"\n=== {len(recs)} specimens with images + GT ===")
    print("by institution:", dict(inst.most_common()))
    print("by state:", dict(state.most_common()))
    fields = {"recordedBy": 0, "county": 0, "locality": 0, "catalogNumber": 0}
    for r in recs:
        for k in fields:
            if r.get(k):
                fields[k] += 1
    print("field coverage:", {k: f"{v}/{len(recs)}" for k, v in fields.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out-dir", default="transcription/data/gbif-ne-500")
    ap.add_argument("--states", nargs="+", default=NEW_ENGLAND)
    # Bias toward the handwritten-label era (matches the current corpus, which
    # runs 1843-1918); the recent end of the range is mostly typed labels.
    ap.add_argument("--year-lo", type=int, default=1850)
    ap.add_argument("--year-hi", type=int, default=1935)
    ap.add_argument("--existing-dir",
                    default="transcription/data/new-england-samples/output",
                    help="exclude occids already in this GT set")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exclude = load_existing_occids(
        os.path.join(args.existing_dir, "taxons.txt"),
        os.path.join(args.existing_dir, "localities.txt"),
    )
    print(f"excluding {len(exclude)} occids already in the existing set")

    records = fetch_records(args.n, args.states, args.year_lo, args.year_hi, exclude)
    print(f"\nfetched {len(records)} qualifying records; downloading images...")
    ok = download_images(records, args.out_dir)
    recs = write_gt(records, ok, args.out_dir)
    summarize(recs)
    print(f"\nWrote images + GT to {args.out_dir}")
