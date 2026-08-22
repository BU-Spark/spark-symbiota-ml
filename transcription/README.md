# Transcription Scripts

## Overview
- `/data/gt-labels` - ground truth `.txt` files (collector, taxon, geography) for sample images, organized by occurrence id.
- `/data/raw-images` - sample images from GBIF to test with.
- `/results` - results and caches from the different pipelines.
- `/utils/image_utils.py` - resizes images to be processed by vision tools.
- `calibration*.json` - per-pipeline confidence calibration maps. See [docs/confidence.md](../docs/confidence.md).

## Pipelines
Each takes an `image_path` and returns a JSON string.

1. `doc_intelligence.py` - Azure Document Intelligence + gpt-4o-mini.
2. `google_vision.py` - Google Document AI + gpt-4o-mini.
3. `claude_sonnet.py` - Claude reads the image directly, no OCR stage.
4. `tesseract_pipeline.py` - Tesseract baseline, unmaintained.

`envelope.py` converts any pipeline's output to the middleware's flat Darwin Core
object plus a `_confidence` map. `confidence.py` produces the per-field scores.

## Environment
Put a `.env` file in this directory with:

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | Azure and Google pipelines (the extraction step) |
| `AZURE_DOCUMENT_KEY` | Azure pipeline |
| `ANTHROPIC_API_KEY` | Anthropic pipeline |
| `GOOGLE_PROJECT_ID` | Google pipeline |
| `GOOGLE_PROCESSOR_ID` | Google pipeline |
| `GOOGLE_LOCATION` | Google pipeline (default `us`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google pipeline: path to a service-account key file |

Google authenticates with a key file rather than a string. For local development
`gcloud auth application-default login` works instead, and
`GOOGLE_APPLICATION_CREDENTIALS` can be left unset. A container has no human to log
in, so it needs the key mounted — see [docker/docker-compose.yaml](../docker/docker-compose.yaml).

The Azure endpoint is currently hardcoded in `doc_intelligence.py`.

Optional:

| Variable | Effect |
|---|---|
| `CONFIDENCE_ENHANCED=0` | Skip the paid confidence signals. Cheaper, fewer scored fields. |
| `CONFIDENCE_DETAIL=1` | Return a dict per field instead of a single float. |

## Install
```
pip install -r transcription/requirements-doc-int.txt
```
The root `requirements.txt` is a conda export for macOS and is not pip-installable.

## Output
```json
{
  "recordedBy": "C. N. Forbes",
  "location": "Oahu",
  "scientificName": "Clermontia persicifolia Gaud.",
  "eventDate": "1912-05-16",
  "barcode": "00427028",
  "institutionCode": "Smithsonian Institution",
  "confidence": {"recordedBy": 0.86, "location": 0.41, "scientificName": 0.94,
                 "eventDate": 0.90, "barcode": 0.88},
  "_taxonMatchType": "EXACT"
}
```
`confidence` is a probability the field is correct, per
[docs/confidence.md](../docs/confidence.md). A field missing from it has no usable
signal. `scientificName` is corrected to GBIF's accepted name; the original read is
kept in `verbatimScientificName` when it differs.
