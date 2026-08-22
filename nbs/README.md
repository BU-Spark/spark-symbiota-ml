# nbs

Evaluation, calibration, and data-preparation scripts. Everything here runs from the
**repository root**, not from this directory.

Every script says at the top whether it costs money. Most read cached outputs and are
free; the ones that call an API say so.


## Calibration

The confidence maps in `transcription/calibration*.json` are produced here. To rebuild
all of them:

```
python nbs/fit_all_calibration.py
```

**Use that rather than the individual fitters.** Each per-pipeline fitter replaces its
whole map file, so running `fit_recordedby_calibration.py` before them silently drops
its field. `fit_all_calibration.py` encodes the order and prints a coverage check at
the end. `--dry-run` shows the order without running anything.

| Script | Produces |
|---|---|
| `fit_all_calibration.py` | all three maps, in order — **the entry point** |
| `fit_calibration.py` | `calibration.json` (Azure, all five fields) |
| `fit_calibration_anthropic.py` | `calibration_anthropic.json` |
| `fit_calibration_google.py` | `calibration_google.json` |
| `fit_recordedby_calibration.py` | the `recordedBy` entry in any map |
| `build_threshold_table.py` | `transcription/threshold_table.json` |

Maps are fitted conservatively by default: each level ships the lower bound of what
the data supports, not the observed average. `--point-estimate` opts out. Why:
[docs/confidence.md](../docs/confidence.md).


## Measuring

| Script | Answers |
|---|---|
| `model_compare.py` | how accurately does each pipeline read each field |
| `confidence_compare.py` | can the confidence score tell right from wrong (rho, AUC, calibration error) |
| `validate_on_holdout.py` | do the shipped maps hold on specimens never fitted on |
| `out_of_domain_eval.py` | do they hold on specimens unlike the training set |
| `sonnet5_hallucination.py` | invented species, impossible years, ungrounded values |
| `model_agreement.py`, `cross_pipeline.py` | do two readers agree, and is that a usable signal |
| `repeat_runs.py` | how much does the same specimen change between runs |
| `confidence_eval.py`, `confidence_threshold_table.py`, `roi_pareto_eval.py` | older Azure-only evals |

`validate_on_holdout.py` scores `transcription/data/holdout-200`, which is disjoint
from every other set. **Never fit on it.** A holdout is only useful while untouched;
if a map needs fixing, fix it against the training splits and draw a fresh holdout.


## Building the caches

Evaluation runs off cached pipeline output so iteration is free. These build the
caches and **cost money**:

| Script | Cost |
|---|---|
| `ocr_cache.py --pipeline azure\|google\|tesseract` | ~$2.00 per 1,000 |
| `model_run.py --model <claude-model>` | varies; prints $/1,000 when done |
| `self_consistency_cache.py` | ~$2.50 per 1,000 |
| `vision_agreement_test.py` | ~$1.00 per 1,000 |

`--occids-from <dir>` on `ocr_cache.py` restricts a build to the specimens another
cache already covers, so pipelines can be compared on identical sheets.


## Getting specimens

```
python nbs/sample_from_export.py --csv <gbif-export.csv> --n 200 --out nbs/my_occids.txt
python nbs/gbif_refetch_occids.py --occid-file nbs/my_occids.txt --out-dir transcription/data/my-set
```

`sample_from_export.py` streams a GBIF occurrence export and picks a sample stratified
by institution — one herbarium means one label format, so an unstratified draw can
span very few. `--exclude-from` keeps a new set disjoint from existing ones.

Expect about **20% of images to fail to download** (hotlink blocks, dead hosts). Draw
more than you need. An occid list rebuilds its image set at any time:
`pilot_occids.txt` (hand-labelling), `holdout_occids.txt` (validation).


## Hand labelling

```
python nbs/label_tool.py --images transcription/data/hand-50
```

Serves a page for labelling specimen sheets by eye, to get truth independent of
GBIF. Candidates from the pipelines are shown shuffled with their source hidden, so
accepting one cannot favour any pipeline. Writes to `<images>-truth/` in the same
format the evals read:

```
HERBARIA_GT_DIR=transcription/data/hand-50-truth python nbs/model_compare.py
```

`HERBARIA_GT_DIR` overrides the ground-truth directory for every eval.


## Notebooks

Older exploratory work, kept for reference. `eval_results.ipynb` and
`eval_tesseract.ipynb` evaluate early Azure/Google/Tesseract CSV outputs against the
`new-england-samples` ground truth. `glm_ocr.ipynb` is a minimal Ollama demo. They
predate the current caching and scoring scripts and are not maintained.
