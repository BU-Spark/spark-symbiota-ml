"""Adapter: pipeline output -> the downstream middleware's flat DWC envelope.

The Herbaria portal middleware accepts a flat
Darwin-Core object plus a parallel `_confidence` map of floats 0-1. A field
missing from `_confidence` means "confidence unavailable" -> the UI shows the
value with no score bar.

We ship confidence for the five validated fields; institutionCode's confidence
is omitted for now.
"""

from typing import Optional

# our field name -> DWC key used in both the flat object and the _confidence map.
# The flat object stays exactly the middleware's contract (one scientificName);
# the raw read is surfaced in _meta, not as a competing flat field.
FIELD_TO_DWC = {
    "recordedBy": "recordedBy",
    "scientificName": "scientificName",   # GBIF-corrected accepted name (the shown value)
    "eventDate": "eventDate",
    "location": "locality",
    "barcode": "catalogNumber",
    "institutionCode": "institutionCode",
}

# Fields whose confidence we currently ship. institutionCode omitted for now.
CONFIDENCE_FIELDS = ["recordedBy", "scientificName", "eventDate", "location", "barcode"]

# Per-field notes surfaced in _meta. Only the Azure pipeline splits location into
# parts, so only it can point at _meta.location_structured.
_LOCALITY_NOTE = ("bundles locality, county and stateProvince; county/state may be "
                  "inferred from the town")
FIELD_NOTES = {
    "locality": _LOCALITY_NOTE + " -- see _meta.location_structured for the parts",
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

    structured = result.get("_locationStructured")
    meta = {
        "model": model,
        "schema_version": SCHEMA_VERSION,
        "field_notes": dict(FIELD_NOTES) if structured
                       else {"locality": _LOCALITY_NOTE},
    }
    # The raw read and match type ride in _meta rather than as flat fields, so the
    # contract shape is unchanged. EXACT means the verbatim was a valid name or a
    # synonym; FUZZY means it was a misread.
    if result.get("verbatimScientificName"):
        meta["verbatim_scientificName"] = result["verbatimScientificName"]
    if result.get("_taxonMatchType"):
        meta["taxon_match_type"] = result["_taxonMatchType"]
    # location is shipped as one flat "locality, county, state" string (DWC locality);
    # the parsed admin parts ride in _meta so the portal can migrate to separate
    # stateProvince/county DWC fields without a pipeline change.
    if structured:
        meta["location_structured"] = structured
    env["_meta"] = meta
    return env
