"""Per-field confidence for the OCR pipelines.

Each field is scored by a reference check: compare the extracted value against an
external authority (GBIF backbone, place gazetteer, known-collector list) or against
an independent second read. OCR grounding is only a fallback -- on its own it does
not predict correctness well.

The extra inputs (sc_samples, vision_fields, taxon_matcher) are optional and enable
the strongest signal for their field. Without them a field degrades to the best
signal available from OCR alone (eventDate -> date validity, scientificName ->
grounding). Default output is one float per field; detail mode adds a dict.

Per-field signals, calibration, and the runbook: docs/confidence.md
"""

import datetime
import json
import os
import re
import unicodedata
import warnings
from difflib import SequenceMatcher
from typing import Optional, Tuple

# The six extracted fields, in output order.
FIELDS = [
    "recordedBy",
    "location",
    "scientificName",
    "eventDate",
    "barcode",
    "institutionCode",
]

_MATCH_THRESHOLD = 0.8   # a field token must match an OCR word at least this closely
_MIN_TOKEN_LEN = 2       # drop shorter tokens (initials, "de", "var") 

# --------------------------------------------------------------------------- #
#  low-level text helpers (also imported by the nbs/ eval scripts)
# --------------------------------------------------------------------------- #
def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def _tokenize(text: str) -> list:
    folded = _strip_accents(str(text).lower())
    return [t for t in re.split(r"\W+", folded) if len(t) >= _MIN_TOKEN_LEN]


def _digits(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def _empty(value: str) -> bool:
    return not value or str(value).strip().upper() == "UNKNOWN"


def _best_match_conf(token: str, word_confs: list) -> Optional[float]:
    # Confidence of the OCR word best matching `token` (exact match for digits).
    if token.isdigit():
        for content, conf in word_confs:
            if content == token:
                return conf
        return None
    best_conf, best_ratio = None, _MATCH_THRESHOLD
    for content, conf in word_confs:
        r = SequenceMatcher(None, token, content).ratio()
        if r >= best_ratio:
            best_ratio, best_conf = r, conf
    return best_conf


def match_field(field_value: str, ocr_words: list) -> Tuple[Optional[float], Optional[float]]:
    # One best-match pass -> (coverage, ocr_read). coverage = fraction of the
    # field's tokens (by length) grounded in OCR; ocr_read = length-weighted mean
    # OCR confidence of matched words. (None, None) for empty/UNKNOWN or no words.
    if _empty(field_value):
        return None, None
    word_confs = [(_strip_accents(str(w["content"]).lower()), w["confidence"])
                  for w in ocr_words
                  if w.get("content") and w.get("confidence") is not None]
    if not word_confs:
        return None, None
    tokens = _tokenize(field_value)
    if not tokens:
        return None, None
    total = matched = weighted = 0.0
    for token in tokens:
        w = len(token)
        total += w
        conf = _best_match_conf(token, word_confs)
        if conf is not None:
            matched += w
            weighted += w * conf
    coverage = round(matched / total, 4)
    ocr_read = round(weighted / matched, 4) if matched else None
    return coverage, ocr_read


def map_field_confidence(field_value: str, ocr_words: list) -> Optional[float]:
    # Grounding score (coverage) -- retained for callers / comparison.
    return match_field(field_value, ocr_words)[0]


# --------------------------------------------------------------------------- #
#  lazy references (offline gazetteers + GBIF taxonomic backbone)
# --------------------------------------------------------------------------- #
_GEO = None            # (us_cities, us_states) lowercased sets, or False if unavailable
_COLLECTORS = None     # set of normalized known-collector full names

# Corpora the known-collector gazetteer is distilled from (relative to this file).
_COLLECTOR_FILES = [
    os.path.join(os.path.dirname(__file__), "data", "gt-labels", "collector_gt.txt"),
    os.path.join(os.path.dirname(__file__), "data", "new-england-samples", "output", "collectors.txt"),
]


_GAZETTEER_WARNED = False


def _warn_no_gazetteer(exc):
    # Without the gazetteer, location confidence degrades to OCR grounding, which
    # does not predict correctness. Warn once so the degradation is visible.
    global _GAZETTEER_WARNED
    if not _GAZETTEER_WARNED:
        _GAZETTEER_WARNED = True
        warnings.warn(
            f"geonamescache unavailable ({exc!r}): location confidence is falling "
            "back to OCR grounding, which is anti-correlated with accuracy. "
            "Install it with: pip install -r transcription/requirements-doc-int.txt",
            RuntimeWarning, stacklevel=3)


def _geo():
    global _GEO
    if _GEO is None:
        try:
            import geonamescache
            g = geonamescache.GeonamesCache()
            cities = {c["name"].lower() for c in g.get_cities().values()
                      if c.get("countrycode") == "US"}
            states = set()
            for s in g.get_us_states().values():
                states |= {s["name"].lower(), s["code"].lower()}
            _GEO = (cities, states)
        except Exception as e:
            _warn_no_gazetteer(e)
            _GEO = False  # geonamescache not installed -> location falls back
    return _GEO


def _collector_fulls():
    global _COLLECTORS
    if _COLLECTORS is None:
        fulls = set()
        for path in _COLLECTOR_FILES:
            if os.path.exists(path):
                for line in open(path, encoding="utf-8", errors="ignore"):
                    nm = line.split(":", 1)[1] if ":" in line else line
                    toks = [t for t in _tokenize(nm) if len(t) >= 3]
                    if toks:
                        fulls.add(" ".join(sorted(toks)))
        _COLLECTORS = fulls
    return _COLLECTORS


class GbifTaxonMatcher:
    # GBIF taxonomic-backbone lookup, cached in-memory + on-disk so repeat names
    # are free. Powers two things: the scientificName confidence signal (match
    # type) and canonical-name correction
    # Returns None match on network error so callers fall back to grounding.
    def __init__(self, cache_file=None, timeout=15):
        self.cache_file = cache_file or os.path.join(
            os.path.dirname(__file__), "results", "gbif_taxon_full.json")
        self.timeout = timeout
        self._cache = {}
        if os.path.exists(self.cache_file):
            try:
                import json
                raw = json.load(open(self.cache_file, encoding="utf-8"))
                # accept legacy {name: "EXACT"} or {name: {"matchType","species"}}
                self._cache = {k: (v if isinstance(v, dict) else {"matchType": v, "species": None})
                               for k, v in raw.items()}
            except Exception:
                self._cache = {}

    @staticmethod
    def _lower_epithet(name) -> str:
        # Old labels capitalise the species epithet ("Rumex Acetosella"), which
        # GBIF does not match.
        parts = str(name).split()
        if len(parts) >= 2 and parts[1][:1].isupper() and parts[1].isalpha():
            parts[1] = parts[1].lower()
            return " ".join(parts)
        return ""

    def match(self, name) -> Optional[dict]:
        # {"matchType": ..., "species": <canonical binomial or None>}; None on error.
        if not name:
            return None
        if name in self._cache:
            res = self._cache[name]
            if res and res.get("matchType") == "NONE":
                alt = self._lower_epithet(name)
                if alt:
                    retry = self.match(alt)
                    if retry and retry.get("matchType") != "NONE":
                        return retry
            return res
        import json
        import urllib.parse
        import urllib.request
        url = "https://api.gbif.org/v1/species/match?name=" + urllib.parse.quote(str(name))
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                d = json.load(r)
            res = {"matchType": d.get("matchType"), "species": d.get("species")}
        except Exception:
            return None  # don't cache transient failures
        self._cache[name] = res
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            json.dump(self._cache, open(self.cache_file, "w", encoding="utf-8"))
        except Exception:
            pass
        if res.get("matchType") == "NONE":
            alt = self._lower_epithet(name)
            if alt:
                retry = self.match(alt)
                if retry and retry.get("matchType") != "NONE":
                    return retry
        return res

    def __call__(self, name) -> Optional[str]:
        m = self.match(name)
        return m.get("matchType") if m else None

    def correct(self, name):
        # (canonical species, match_type). Replaces the read with GBIF's accepted
        # species on EXACT (already-valid or synonym) or FUZZY (misread) matches;
        # otherwise returns the name unchanged.
        m = self.match(name)
        if m and m.get("matchType") in ("EXACT", "FUZZY") and m.get("species"):
            return m["species"], m["matchType"]
        return name, (m.get("matchType") if m else None)


def _sc_agreement(values: list) -> Optional[float]:
    # Graded self-consistency: mean pairwise string similarity across K samples.
    vals = [str(v).strip().lower() for v in values if str(v).strip()]
    if len(vals) < 2:
        return None
    sims = [SequenceMatcher(None, vals[i], vals[j]).ratio()
            for i in range(len(vals)) for j in range(i + 1, len(vals))]
    return sum(sims) / len(sims)


# --------------------------------------------------------------------------- #
#  per-field confidence scorers (validated best signal + graceful fallback)
# --------------------------------------------------------------------------- #
_STATE_IDX = None  # {state name or 2-letter code (lower) -> canonical 2-letter code}

# Traditional state abbreviations as they appear on historic labels
# ("north Branford, Conn."). geonamescache carries only full names and 2-letter
# codes. Forms that collide with common words ("ok", "in", "or", "me") are
# omitted; those already arrive as 2-letter codes.
_STATE_ABBREV = {
    "ala": "AL", "ariz": "AZ", "ark": "AR", "cal": "CA", "calif": "CA",
    "colo": "CO", "conn": "CT", "del": "DE", "fla": "FL", "ida": "ID",
    "ill": "IL", "ind": "IN", "kan": "KS", "kans": "KS", "mass": "MA",
    "mich": "MI", "minn": "MN", "miss": "MS", "mont": "MT", "neb": "NE",
    "nebr": "NE", "nev": "NV", "okla": "OK", "ore": "OR", "oreg": "OR",
    "penn": "PA", "penna": "PA", "tenn": "TN", "tex": "TX", "vt": "VT",
    "wash": "WA", "wis": "WI", "wisc": "WI", "wyo": "WY",
    # dotted multi-word forms, post-normalisation (see _strip_letter_dots)
    "nh": "NH", "nj": "NJ", "nm": "NM", "ny": "NY", "nc": "NC", "nd": "ND",
    "ndak": "ND", "ri": "RI", "sc": "SC", "sd": "SD", "sdak": "SD",
    "wva": "WV", "dc": "DC",
}

# "N.H." -> "NH", "W. Va." -> "WVa". Strips a period following a single-letter
# word so the split on [,;.] does not break dotted abbreviations apart.
_LETTER_DOT_RE = re.compile(r"(?<=\b[A-Za-z])\.\s?")


def _strip_letter_dots(text: str) -> str:
    return _LETTER_DOT_RE.sub("", str(text))


def _state_index():
    global _STATE_IDX
    if _STATE_IDX is None:
        idx = {}
        try:
            import geonamescache
            for s in geonamescache.GeonamesCache().get_us_states().values():
                idx[s["name"].lower()] = s["code"]
                idx[s["code"].lower()] = s["code"]
        except Exception as e:
            _warn_no_gazetteer(e)
        # Static, so state parsing still works without the gazetteer.
        for k, v in _STATE_ABBREV.items():
            idx.setdefault(k, v)
        _STATE_IDX = idx
    return _STATE_IDX


def _state_code(text: str) -> Optional[str]:
    # Canonical US state code parsed from a location string ("Wolfeboro, Carroll,
    # New Hampshire" -> "NH"), so an OCR-derived state and a vision-derived state
    # can be compared regardless of name/abbreviation form.
    idx = _state_index()
    if not idx or _empty(text):
        return None
    # Our flat location is "locality, county, state" -> the state is the LAST
    # component. Scan components from the end so a state-named locality/county
    # ("Washington, Litchfield, Connecticut") doesn't shadow the real state.
    parts = [p.strip() for p in re.split(r"[,;.]", _strip_letter_dots(text)) if p.strip()]
    for part in reversed(parts):
        if part.lower() in idx:
            return idx[part.lower()]
    for part in reversed(parts):
        for t in _tokenize(part):
            if t in idx:
                return idx[t]
    return None


def _location_completeness(value: str, ocr_words: list = None) -> Optional[float]:
    # Fraction of recognized real places (town/county/state), capped at 3. Weak
    # fallback used when a state can't be parsed for the vision cross-check.
    geo = _geo()
    if not geo:
        return match_field(value, ocr_words or [])[0]
    cities, states = geo
    comps = [c.strip() for c in re.split(r"[,;.]", str(value)) if c.strip()]
    if not comps:
        return None
    known = sum(1 for c in comps if c.lower() in cities or c.lower() in states
                or any(t in cities or t in states for t in _tokenize(c)))
    return round(min(known, 3) / 3.0, 4)


def location_confidence(value: str, ocr_words: list = None,
                        vision_value: str = None) -> Optional[float]:
    # Graded place count is the score; a vision read that parses a different state
    # halves it. State agreement is too coarse to be the score on its own -- it
    # says nothing about the locality and county, which carry most of the accuracy.
    if _empty(value):
        return None
    base = _location_completeness(value, ocr_words)
    if base is None:
        return None
    vis_st = _state_code(vision_value) if vision_value and not _empty(vision_value) else None
    if vis_st is not None and _state_code(value) != vis_st:
        base *= 0.5
    return round(base, 4)


def _sig_digits(text: str) -> str:
    # Catalog-number digits without leading-zero padding, so "CBS.24501" and the
    # DB's zero-padded "CBS.024501" compare equal 
    return _digits(text).lstrip("0")


def _barcode_sc_agreement(value: str, sc_samples: list) -> Optional[float]:
    # Fraction of the self-consistency re-reads whose catalog number matches (modulo
    # leading zeros). A stably-reproduced number is trustworthy; a single-digit
    # misread flips across re-reads and scores low.
    sv = _sig_digits(value)
    if not sv:
        return None
    vals = [_sig_digits(s) for s in sc_samples if s]
    vals = [v for v in vals if v]
    if not vals:
        return None
    return round(sum(1 for v in vals if v == sv) / len(vals), 4)


def barcode_confidence(value: str, ocr_words: list, sc_samples: list = None) -> Optional[float]:
    # SELF-CONSISTENCY: how stably the K
    # re-reads reproduce the same catalog number. A shaky digit flips across re-reads.
    # Falls back to the length ratio when no self-consistency context.
    if _empty(value):
        return None
    d = _digits(value)
    if not d:
        return 0.0
    if sc_samples:
        agree = _barcode_sc_agreement(value, sc_samples)
        if agree is not None:
            return agree
    runs = [x for x in (_digits(w["content"]) for w in ocr_words if w.get("content")) if x]
    if not runs:
        return 0.5
    return round(min(len(d) / max(len(x) for x in runs), 1.0), 4)


def scientificname_confidence(value: str, ocr_words: list = None,
                              taxon_matcher=None, vision_value: str = None) -> Optional[float]:
    # GBIF backbone match (EXACT->1.0, FUZZY->0.5, else 0.0), ensembled with the
    # independent VISION read agreement. Name-match alone saturates; multiplying by vision
    # agreement pulls disagreements down and cuts confident-wrong
    if _empty(value):
        return None
    name = None
    if taxon_matcher is not None:
        mt = taxon_matcher(value)
        if mt is not None:
            name = {"EXACT": 1.0, "FUZZY": 0.5}.get(mt, 0.0)
    if name is None:
        name = match_field(value, ocr_words or [])[0]
    if name is not None and vision_value and not _empty(vision_value):
        return round(name * _ratio(value, vision_value), 4)
    return name


def recordedby_confidence(value: str, ocr_words: list = None,
                          vision_value: str = None) -> Optional[float]:
    #  fuzzy match to the known-collector set, ensembled (mean) with
    # the independent vision read. Falls back to whichever is available.
    if _empty(value):
        return None
    fulls = _collector_fulls()
    nj = " ".join(sorted(t for t in _tokenize(value) if len(t) >= 3))
    ref = max((_ratio(nj, k) for k in fulls), default=0.0) if (nj and fulls) else None
    vis = _ratio(value, vision_value) if vision_value and not _empty(vision_value) else None
    parts = [p for p in (ref, vis) if p is not None]
    if parts:
        return round(sum(parts) / len(parts), 4)
    return match_field(value, ocr_words or [])[0]


_YEAR_RE = re.compile(r"(1[6-9]\d\d|20[0-2]\d)")


def _event_year(text: str) -> Optional[int]:
    m = _YEAR_RE.search(str(text))
    return int(m.group(1)) if m else None


def _is_valid_date(text: str) -> bool:
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            datetime.datetime.strptime(str(text).strip(), fmt)
            return True
        except ValueError:
            pass
    return False


def _eventdate_validity(value: str, ocr_words: list) -> Optional[float]:
    # Fallback when no self-consistency / vision context: validity + plausibility
    # + year-grounding. Weak (rho ~+0.06) but free and better than nothing.
    if _empty(value):
        return None
    year = _event_year(value)
    if not _is_valid_date(value) or year is None or not (1600 <= year <= datetime.date.today().year):
        return 0.0
    ys = str(year)
    grounded = any(_digits(w["content"]) == ys for w in ocr_words if w.get("content"))
    return 1.0 if grounded else 0.6


def eventdate_confidence(value: str, ocr_words: list, sc_samples: list = None,
                         vision_value: str = None, k: int = 3) -> Optional[float]:
    # Base = mean of self-consistency agreement (K re-reads) + vision string
    # agreement; falls back to date validity when neither is supplied. Then two
    # corrections that cut confident-wrong 16%->~10% 
    if _empty(value):
        return None
    parts = []
    if sc_samples:
        a = _sc_agreement(sc_samples[:k])
        if a is not None:
            parts.append(a)
    if vision_value and not _empty(vision_value):
        parts.append(_ratio(value, vision_value))
    score = round(sum(parts) / len(parts), 4) if parts else _eventdate_validity(value, ocr_words)
    if score is None:
        return None
    py = _event_year(value)
    if py is None or not (1600 <= py <= datetime.date.today().year):
        return 0.0  # implausible year -> not trustworthy regardless of agreement
    if vision_value and not _empty(vision_value):
        vy = _event_year(vision_value)
        if vy is not None and vy != py:
            score *= 0.5
    return round(score, 4)


# --------------------------------------------------------------------------- #
#  calibration (raw signal -> probability of correctness)
# --------------------------------------------------------------------------- #
# {map name: {field: [[x, p], ...]}} isotonic knots. Each pipeline has its own map.
_CALIBRATION = {}
DEFAULT_CALIBRATION = "calibration"


def _calibration(name: str = DEFAULT_CALIBRATION):
    if name not in _CALIBRATION:
        path = os.path.join(os.path.dirname(__file__), f"{name}.json")
        try:
            _CALIBRATION[name] = json.load(open(path, encoding="utf-8"))
        except Exception:
            _CALIBRATION[name] = {}  # no map -> scores pass through uncalibrated
    return _CALIBRATION[name]


def _apply_calibration(field: str, score: Optional[float],
                       name: str = DEFAULT_CALIBRATION) -> Optional[float]:
    # Piecewise-linear between the field's isotonic knots, so conf=0.8 means ~80%
    # correct. No map, or name=None, returns the raw signal unchanged.
    if score is None or not name:
        return score
    knots = _calibration(name).get(field)
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
            return round(y0 if x1 == x0 else y0 + (y1 - y0) * (score - x0) / (x1 - x0), 4)
    return knots[-1][1]


# --------------------------------------------------------------------------- #
#  assembly
# --------------------------------------------------------------------------- #
def _field_confidence(field: str, value: str, ocr_words: list,
                      sc_samples, vision_fields, taxon_matcher):
    v = lambda f: (vision_fields or {}).get(f)  # noqa: E731
    if field == "location":
        return location_confidence(value, ocr_words, v("location"))
    if field == "barcode":
        bscs = [s.get("barcode", "") for s in sc_samples] if sc_samples else None
        return barcode_confidence(value, ocr_words, bscs)
    if field == "scientificName":
        return scientificname_confidence(value, ocr_words, taxon_matcher, v("scientificName"))
    if field == "recordedBy":
        return recordedby_confidence(value, ocr_words, v("recordedBy"))
    if field == "eventDate":
        scs = [s.get("eventDate", "") for s in sc_samples] if sc_samples else None
        return eventdate_confidence(value, ocr_words, scs, v("eventDate"))
    if field == "institutionCode":
        return None  # deferred: no valid ground truth, not in middleware
    return match_field(value, ocr_words)[0]


def build_confidence(fields: dict, ocr_words: list, sc_samples: list = None,
                     vision_fields: dict = None, taxon_matcher=None,
                     detail: bool = False,
                     calibration: str = DEFAULT_CALIBRATION) -> dict:
    # Per-field confidence object. Optional context enables the strong signals:
    #   sc_samples     -- list of K self-consistency extraction dicts (eventDate)
    #   vision_fields  -- an independent vision read's field dict (eventDate, recordedBy)
    #   taxon_matcher  -- callable name->GBIF matchType (scientificName)
    #   calibration    -- calibration map name for this pipeline
    # Default output: {field: float|None}. detail=True: {field: {"score":..., "coverage":...}}.
    confidence = {}
    for field in FIELDS:
        raw = _field_confidence(field, fields.get(field, ""), ocr_words,
                                sc_samples, vision_fields, taxon_matcher)
        # Calibrate the raw signal to a probability of correctness (monotonic, so
        # ranking is unchanged; only the numbers become meaningful). No-op for
        # fields without a calibration map or when the map file is absent.
        score = _apply_calibration(field, raw, calibration)
        if detail:
            # keep a "coverage" key so envelope.py's detail-shape fallback works
            confidence[field] = {"score": score, "coverage": score, "raw": raw,
                                 "ocr_read": match_field(fields.get(field, ""), ocr_words)[1]}
        else:
            confidence[field] = score
    return confidence


def detail_enabled() -> bool:
    return os.environ.get("CONFIDENCE_DETAIL", "").lower() in ("1", "true", "yes")


# Retained for callers importing it (the old barcode format pattern).
_BARCODE_PATTERN = re.compile(r"^[A-Za-z]{0,5}[.\-_ ]?\d{3,}$")
