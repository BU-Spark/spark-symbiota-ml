"""Rebuild transcription/data/gbif-ne-500 for the occids the caches are keyed on.

nbs/gbif_fetch.py builds a new set via occurrence/search, which returns different
occids and orphans the existing caches. This fetches the exact occids already in
the caches by key, so the restored images and ground truth line up with what is on
disk. Use it when the caches are present but the image set is not.

    python nbs/gbif_refetch_occids.py                 # ground truth + images
    python nbs/gbif_refetch_occids.py --skip-images   # ground truth only

Free: GBIF is a public API. Idempotent -- existing images are skipped.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from gbif_fetch import USER_AGENT, download_images, summarize, write_gt  # noqa: E402

OCCURRENCE = "https://api.gbif.org/v1/occurrence/"


def occids_from_caches(*cache_dirs) -> list:
    """Union of occids across the given cache dirs (filenames are <occid>.json)."""
    occ = set()
    for d in cache_dirs:
        if os.path.isdir(d):
            occ.update(f[:-5] for f in os.listdir(d) if f.endswith(".json"))
    return sorted(occ)


def fetch_one(occid: str, retries: int = 3) -> dict:
    req = urllib.request.Request(OCCURRENCE + occid,
                                 headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}          # record withdrawn from GBIF
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    return {}


def fetch_records(occids: list, workers: int = 8) -> list:
    recs, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, o): o for o in occids}
        for i, fut in enumerate(as_completed(futs), 1):
            occid = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                failed.append((occid, str(e)[:80]))
                continue
            if rec.get("key"):
                recs.append(rec)
            else:
                failed.append((occid, "not found"))
            if i % 100 == 0:
                print(f"  fetched {i}/{len(occids)}", flush=True)
    for occid, msg in failed:
        print(f"  REC FAIL {occid}: {msg}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="transcription/data/gbif-ne-500")
    ap.add_argument("--cache-dirs", nargs="+", default=[
        "transcription/results/ocr_cache/azure",
        "transcription/results/sonnet5_cache",
    ])
    ap.add_argument("--occid-file",
                    help="file of occids, one per line (from nbs/sample_from_export.py); "
                         "overrides --cache-dirs")
    ap.add_argument("--skip-images", action="store_true",
                    help="write ground truth only; don't download the scans")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.occid_file:
        occids = [l.strip() for l in open(args.occid_file, encoding="utf-8") if l.strip()]
        src = args.occid_file
    else:
        occids = occids_from_caches(*args.cache_dirs)
        src = str(args.cache_dirs)
    if not occids:
        raise SystemExit(f"no occids found in {src}")
    print(f"{len(occids)} occids from {src}; fetching GBIF records...", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    records = fetch_records(occids, args.workers)
    print(f"\nresolved {len(records)}/{len(occids)} records")

    if args.skip_images:
        ok = {str(r["key"]) for r in records}
        print("--skip-images: writing GT for all resolved records")
    else:
        print("downloading images...", flush=True)
        ok = download_images(records, args.out_dir)

    recs = write_gt(records, ok, args.out_dir)
    summarize(recs)
    print(f"\nWrote GT{'' if args.skip_images else ' + images'} to {args.out_dir}")


if __name__ == "__main__":
    main()
