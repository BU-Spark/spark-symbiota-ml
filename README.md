# spark-symbiota-ml
Source code for models developed by BU Spark to enable plant specimen recognition for Symbiota

## Directories
1. transcription: Contains the transcription scripts for performing ocr.
2. backend: Contains a simple api implementation with one route to receive the url of an image, downloads and verifies the image, and then calls the transcription scripts to perform ocr on the downloaded image.
3. docker: Contains a docker compose implementation of the backend server.

## How to run the ocr service
1. Put the `.env` file containing the required variables (see the [transcription README](/transcription/README.md))in the transcription directory.
2. To do a minimal installation of only the Azure pipeline, install the packages in `transcription/requirements-doc-int.txt` and then `backend/requirements-backend.txt` in order. Otherwise, you can install the root `requirements.txt` and then `backend/requirements-backend.txt`. 
3. Navigate to the `docker` directory and run `docker compose up -d`.