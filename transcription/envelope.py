"""Adapter: pipeline output -> the downstream middleware's flat DWC envelope.

The Herbaria portal middleware (see the SWE team's contract) accepts a flat
Darwin-Core object plus a parallel `_confidence` map of floats 0-1. A field
missing from `_confidence` means "confidence unavailable" -> the UI shows the
value with no score bar.

We ship confidence for the five validated fields; institutionCode's confidence
is omitted for now (no valid ground truth yet -- being resolved with the team),
and locality is flagged low-reliability in `_meta` while its scorer is improved.
"""

from typing import Optional

# our field name -> DWC key used in both the flat object and the _confidence map
FIELD_TO_DWC = {
    "recordedBy": "recordedBy",
    "scientificName": "scientificName",
    "eventDate": "eventDate",
    "location": "locality",
    "barcode": "catalogNumber",       # TODO: confirm exact DWC key with SWE team
    "institutionCode": "institutionCode",
}

# Fields whose confidence we currently ship. institutionCode omitted for now.
CONFIDENCE_FIELDS = ["recordedBy", "scientificName", "eventDate", "location", "barcode"]

# Per-field notes surfaced in _meta (e.g. so the UI can mark a weak signal).
FIELD_NOTES = {
    "locality": "low reliability; scorer improvement in progress",
}

SCHEMA_VERSION = 1


def _clamp01(x) -> Optional[float]:
    # Coerce to a float in [0, 1]; drop null / NaN / non-numeric.
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


def to_middleware_envelope(result: dict, model: str = "azure") -> dict:
    # result is the parsed pipeline output: the six fields plus a "confidence"
    # map (single floats by default, or {"coverage","ocr_read"} in detail mode).
    conf = result.get("confidence", {}) or {}

    env = {}
    # Flat DWC fields (only those actually present in the result).
    for our, dwc in FIELD_TO_DWC.items():
        if our in result:
            env[dwc] = result[our]

    # _confidence: single float per shipped field, clamped, nulls dropped.
    confidence = {}
    for our in CONFIDENCE_FIELDS:
        raw = conf.get(our)
        if isinstance(raw, dict):          # detail shape -> take the headline
            raw = raw.get("coverage")
        v = _clamp01(raw)
        if v is not None:
            confidence[FIELD_TO_DWC[our]] = v
    env["_confidence"] = confidence

    env["_meta"] = {
        "model": model,
        "schema_version": SCHEMA_VERSION,
        "field_notes": FIELD_NOTES,
    }
    return env
