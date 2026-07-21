# Notebooks (nbs)

This README documents the primary notebooks used to evaluate OCR outputs and experiment with an Ollama-based OCR model. It summarizes purpose, inputs, outputs, dependencies, and quick run notes for each notebook.

**Notebooks**
- **eval_results.ipynb**: Evaluates OCR outputs from multiple services (Azure, Google). Loads results CSVs (e.g., `transcription/results/azure_doc_intell_results_0.csv`, `transcription/results/google_doc_ai_results.csv`), compares predicted `scientificName`, `location`, and `eventDate` against ground-truth text files, computes Levenshtein similarity scores, produces accuracy counts by thresholds, and performs post-processing (e.g., name matching with `wcvpy`). Writes appended CSVs to `transcription/results/*_appended.csv`.

  Inputs:
  - Ground truth text files: `transcription/data/new-england-samples/output/dates.txt`, `localities.txt`, `taxons.txt`
  - OCR results CSVs: `transcription/results/*` (Azure/Google outputs)

  Outputs:
  - Appended CSVs with similarity scores and cleaned fields (written to `transcription/results/`)

  Key libs used: `pandas`, `dateutil`, `Levenshtein`, `wcvpy` (optional postprocessing), `numpy`.

- **eval_tesseract.ipynb**: Focused evaluation of Tesseract OCR outputs. Reads Tesseract results CSV (`transcription/results/tesseract_results.csv`), extracts candidate taxon names using regex, validates via GBIF API, extracts locations & dates via spaCy NER, chooses best candidate per field using Levenshtein ratio, appends predictions and similarity scores to the dataframe, computes accuracy metrics, and provides helper tooling to review images for high-confidence matches.

  Inputs:
  - `transcription/results/tesseract_results.csv`
  - Ground truth text files as above

  Outputs:
  - Dataframe with `Actual` and `Predicted` columns, Levenshtein ratios, optional CSV export (example path in notebook: `transcription/results/tesseract_results_appended_0.csv`).

  Key libs used: `spaCy` (en_core_web_sm), `requests` (GBIF), `pandas`, `Levenshtein`, `Pillow`.

  Notes:
  - Taxon extraction uses a simple regex for binomial names and a GBIF fuzzy match; consider using a dedicated taxon NER for improved recall.
  - Thresholds for accuracy checks in the notebook are configurable (examples use 0.7 for many checks).

- **glm_ocr.ipynb**: Minimal demo showing use of `ollama.chat` with a local `glm-ocr` model to extract text (and optionally structured fields) from a specimen image. Example shows sending an image path to the model and printing the response. The notebook includes a commented prompt variant that requests JSON output for collector, location, scientific name, event date, barcode, and institution code.

  Inputs:
  - Local image file path (example: `transcription/data/new-england-samples/output/<occid>.jpeg`)

  Outputs:
  - Model response printed to stdout; can be adapted to save JSON to CSV.

  Requirements:
  - Ollama and the `glm-ocr` model installed and available locally (see Ollama docs)
  - Python package `ollama` available to the notebook environment

**Quick setup & run**
- Open these notebooks in JupyterLab / Jupyter Notebook.
- Install Python dependencies (many are defined in repository requirements). At minimum, ensure:
  - `pandas`, `numpy`, `python-dateutil`, `spacy`, `Levenshtein` (python-Levenshtein), `Pillow`, `requests`.
  - For `eval_tesseract.ipynb`: install and download spaCy model `en_core_web_sm`.
  - For `glm_ocr.ipynb`: install `ollama` and ensure the `glm-ocr` model is present.


**Paths & outputs**
- Ground truth: `transcription/data/new-england-samples/output/` (`dates.txt`, `localities.txt`, `taxons.txt`)
- OCR results: `transcription/results/` (various service outputs and appended CSVs)
- Images: `transcription/data/new-england-samples/output/<occid>.jpeg`

**Notes & next steps**
- Notebooks assume relative paths from the repository root (open the notebooks with the repo root as working directory).
- `eval_results.ipynb` contains post-processing steps (name matching, acceptance via `wcvpy`) that can increase taxon-match rates; these are optional but useful for finalizing results.
- Consider replacing regex taxon extraction with a trained taxon NER for better recall.

