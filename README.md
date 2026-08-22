# spark-symbiota-ml
Source code for models developed by BU Spark to enable plant specimen recognition for Symbiota

## Directories
1. `transcription`: the OCR/transcription pipelines and the confidence layer.
2. `backend`: the API. One route per pipeline, plus a browser tool for comparing them. See [backend/README.md](/backend/README.md).
3. `docker`: docker compose implementation of the backend server.
4. `nbs`: evaluation and calibration scripts. See [nbs/README.md](/nbs/README.md).
5. `docs`: measurements, cost, and how confidence works.

## Pipelines
Three, all returning the same Darwin Core envelope, so they are interchangeable.

| Pipeline | How it reads the sheet | Cost per 1,000 |
|---|---|---|
| `azure` | Azure Document Intelligence OCR + gpt-4o-mini | $4.50 |
| `google` | Google Document AI + gpt-4o-mini | $2.00 |
| `anthropic` | Claude reads the image directly | $19.90 |

Accuracy and the pipeline recommendation: [docs/recommendation.md](/docs/recommendation.md).

## Running the ocr service
1. Put the `.env` file containing the required variables (see the [transcription README](/transcription/README.md)) in the transcription directory.
2. Install `transcription/requirements-doc-int.txt` and then `backend/requirements-backend.txt`, in that order. The root `requirements.txt` is a conda export for macOS and is not pip-installable.
3. If the docker network `symbiota-network` has not been created yet, run `docker network create symbiota-network`. The ocr service and the ocr middleware can be hosted on this network together. Change this in `docker/docker-compose.yaml` according to your hosting preferences.
4. Set `GOOGLE_SA_KEY` to the path of a Google service-account key file. Compose refuses to start without it, because Google authenticates with a file rather than an environment string. Only the `google` pipeline needs it.
5. Navigate to the `docker` directory and run `docker compose up -d`.

Then `http://localhost:8080/` returns the service status, and `http://localhost:8080/tool`
opens the comparison tool.
