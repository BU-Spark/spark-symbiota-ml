# backend

FastAPI service in front of the transcription pipelines. One route per pipeline, all
returning the same Darwin Core envelope, plus a browser tool for comparing them.

```
python -m uvicorn backend.main:app --port 8080
```

Or via docker compose — see the [root README](../README.md).


## Routes

| Route | |
|---|---|
| `GET /` | service status. **Original contract — the middleware may health-check this** |
| `GET /health` | same payload, for anything that prefers an explicit name |
| `POST /azure`, `/google`, `/anthropic` | transcribe one image |
| `POST /compare` | run several pipelines over one upload |
| `GET /pipelines` | available pipelines and their per-field confidence ceilings |
| `GET /tool` | the browser tool |
| `POST /reviews`, `GET /reviews` | reviewer corrections |

Every transcribe route takes either `?url=` or a multipart `file`:

```
curl -X POST "localhost:8080/google?url=https://example.org/sheet.jpg"
curl -X POST localhost:8080/google -F "file=@sheet.jpg"
```

Uploads are capped at 40MB and must be images. `POST /compare` takes
`?pipelines=azure,google` and runs one upload through each, so a large sheet is not
sent three times.


## Response

Flat Darwin Core keys plus a parallel `_confidence` map of probabilities, and `_meta`:

```json
{
  "scientificName": "Rumex acetosella",
  "recordedBy": "J. Blake",
  "eventDate": "1864-07-06",
  "locality": "Wells, Maine",
  "catalogNumber": "372522",
  "institutionCode": "F",
  "_confidence": {"scientificName": 0.94, "recordedBy": 0.86, "eventDate": 0.90,
                  "locality": 0.41, "catalogNumber": 0.88},
  "_meta": {"model": "azure", "schema_version": 1,
            "verbatim_scientificName": "Rumex Acetosella",
            "taxon_match_type": "EXACT"}
}
```

A field missing from `_confidence` has no usable signal — render the value without a
score rather than as a low one. What the numbers mean:
[docs/confidence.md](../docs/confidence.md).

The shape is produced by `transcription/envelope.py` and is identical across
pipelines, which is what makes them interchangeable behind a selector.


## The tool

`GET /tool` serves a page for uploading a sheet, running one or all pipelines, and
seeing the fields side by side with their confidence. It is for **choosing** a
pipeline and a review threshold, not for production digitisation — averages measured
on our specimens cannot tell a collection what it will get on theirs.

Corrections typed there are appended to a review log. Each row is a labelled specimen
from that collection, which is the input a per-collection calibration would need.
Nothing reads the log yet.


## Deploying

**Google needs a key file.** It authenticates with a file rather than an environment
string, so `GOOGLE_APPLICATION_CREDENTIALS` must point at a mounted service-account
key. Without it `/google` returns 502; the other pipelines are unaffected. Local
development can use `gcloud auth application-default login` instead.

**Set `REVIEWS_PATH`** to a mounted volume if `/tool` is exposed. Unset, reviews are
written inside the container and lost on restart.

All other credentials come from `transcription/.env` — see the
[transcription README](../transcription/README.md).


## Tests

```
python tests/test_backend_concurrency.py     # or: pytest tests/
```

Covers request isolation (every request gets its own temp file), upload validation,
every pipeline having a wired route, and that `GET /` keeps its JSON contract.


## Known gaps

- No retry or backoff on any upstream call. A transient failure is a lost request.
- Requests are handled one at a time per worker; there is no batching.
- Pipelines return errors as strings rather than raising, so `main.py` inspects the
  result to tell a failure from data.
