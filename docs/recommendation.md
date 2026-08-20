# Pipeline recommendation

What to ship, what it costs, and when to use which. Evidence is in
`docs/lab-notebook.md`; prices are in `docs/cost-projection.md`.

## Recommendation

**Offer both pipelines. Default to Azure for bulk digitisation, and offer Claude
Sonnet 5 for collections where the record will be published or relied on.**

| | Azure | Anthropic |
|---|---|---|
| Extractor | Azure OCR + gpt-4o-mini | Claude Sonnet 5, image-native |
| Accuracy | 74% | **85%** |
| Invented species names | 11.3% | **1.3%** |
| Cost per 1,000 | **$4.50** | $19.90 |
| Confidence | 5 fields, 4 calibrated | 5 fields, 4 calibrated |

Claude costs about 4x more and is 11 accuracy points better. That is a real tradeoff,
not an upgrade, which is why both stay available.

**If only one can be maintained, ship Claude Sonnet 5.** Two of the Azure gaps are
severe enough to affect published data: it reads `location` correctly 55% of the
time, and roughly one in nine scientific names it emits does not exist.

## The options, priced

| Option | Accuracy | $/1,000 | Confidence |
|---|---:|---:|---|
| Azure, extraction only | 74% | $2.00 | none |
| **Azure, full** | **74%** | **$4.50** | 5 fields |
| Claude Sonnet 5, extraction only | 85% | $11.34 | 2 fields |
| **Claude Sonnet 5, full** | **85%** | **$19.90** | 5 fields |

Confidence is priced separately because it is optional, and because it is what turns
accuracy into saved review time. Without it every field needs checking.

## Why Sonnet 5 and not another Claude model

Six models were measured on the same 150 specimens.

| Model | Accuracy | $/1,000 |
|---|---:|---:|
| Opus 4.8 | 85.3% | $25.81 |
| **Sonnet 5** | **84.9%** | **$11.34** |
| Opus 4.6 | 78.6% | $10.78 |
| Sonnet 4.6 | 75.1% | $6.56 |
| Haiku 4.5 | 49.1% | $2.17 |

Sonnet 5 matches the most expensive model tested to within measurement error, at less
than half the price. Opus 4.8 buys 0.4 points for 2.3x the cost.

The cheaper tier is not competitive: `barcode` splits by model generation, with Opus
4.8 and Sonnet 5 reading about 90% and every 4.6-era model about 55%. Haiku 4.5 is
unusable — it reads handwritten dates correctly 39% of the time.

**Note the pricing deadline.** Sonnet 5 is at an introductory rate until 2026-08-31,
after which its cost per 1,000 rises from $11.34 to about $17.51. Re-check the
tradeoff then; it does not change the ranking, but it narrows the gap to Opus 4.8.

## Where each pipeline wins

Per-field accuracy, Sonnet 5 against the current production pipeline:

| Field | Azure | Claude Sonnet 5 |
|---|---:|---:|
| `location` | 54.5% | **85.6%** |
| `scientificName` | 71.4% | **85.3%** |
| `eventDate` | 78.9% | **87.2%** |
| `barcode` | 90.5% | 91.3% |
| `recordedBy` | 75.3% | 75.0% |

The gap is concentrated in `location` and `scientificName`, and both have the same
cause: gpt-4o-mini never sees the specimen. It only receives whatever text Azure's OCR
managed to extract, so when the OCR garbles handwritten cursive it has nothing to
check against and completes the fragment into something plausible. A model looking at
the sheet can read the handwriting.

That also explains the fabrication rate. Names that do not exist in the global species
database:

| Pipeline | Invented species names |
|---|---:|
| Azure (gpt-4o-mini) | 11.3% |
| Claude Sonnet 5 | 1.3% |
| Claude Opus 4.8 | 0.7% |

These are automatically detectable — a name GBIF cannot match at any rank is
fabricated by definition — so they can be flagged for review on either pipeline rather
than shipped silently. That is worth doing regardless of which pipeline is chosen.

## What each pipeline ships with

Both produce the same flat Darwin Core object plus a parallel `_confidence` map, so
they are interchangeable behind a selector.

| Field | Azure signal | Claude signal |
|---|---|---|
| `scientificName` | GBIF match type + vision agreement | GBIF match type |
| `location` | place gazetteer + vision state check | place gazetteer |
| `eventDate` | self-consistency + vision agreement | agreement with a checker model |
| `recordedBy` | collector list + vision agreement | collector list + checker agreement |
| `barcode` | digit-run structure + self-consistency | agreement with the Azure read |

Both have calibrated maps on four of five fields; `recordedBy` ships raw on both,
because its score is not monotonically related to accuracy and calibration makes it
worse. Calibration means a shipped 0.9 is roughly 90% likely to be correct, so the two
pipelines' numbers mean the same thing in the same UI.

## Setting the review threshold

At a confidence cutoff of 0.90 on the Azure pipeline:

| | |
|---|---|
| Fields auto-accepted | 47% |
| Wrong among those | ~5% |
| Saved on a 1.5M corpus | ~7,900 reviewer-hours |

Raising the cutoff above 0.90 buys little — coverage falls while the missed-error rate
stays flat, because calibration compresses the top of the range. 0.90 is a reasonable
default for the rapid-entry tool.

## An improvement available to the Azure pipeline

Azure's confidence currently uses a gpt-4o-mini vision read as its independent check.
Replacing that with a Claude read improves its two weakest fields:

| Field | with gpt-4o-mini | with Claude |
|---|---:|---:|
| `recordedBy` | +0.394 | **+0.535** |
| `eventDate` | +0.363 | **+0.504** |

gpt-4o-mini checking gpt-4o-mini shares failure modes; a different vendor reading a
different input does not. Cost rises from $4.50 to about $10 per 1,000, so this belongs
behind a flag rather than as the default.

## What we are not confident about

**The ground truth is partly circular.** It comes from GBIF, and `scientificName`'s
strongest signal is a GBIF backbone match, so that field is partly graded by its own
source. The hand-labelled non-GBIF test set in `cost-projection.md` part (b) is the fix,
and it is cheap — about $9.

**The model comparison is n=150.** Differences under about 3 points are inside noise.
The 10-point gaps are safe; the 0.4-point gap between Sonnet 5 and Opus 4.8 is not
meaningful.

**Neither pipeline is reproducible run to run.** Five runs of the same specimen change
at least one field on 40% of specimens — usually formatting or completeness, sometimes
a genuine disagreement. Aggregating five runs does not improve accuracy, so the answer
is to store results rather than regenerate them on demand.

**`location` ground truth covers about two-thirds of specimens**, so that column rests
on a smaller sample than the others.

## Before this ships

| | |
|---|---|
| Top up the OpenAI credit | the Azure pipeline cannot run at all right now, and it supplies Claude's `barcode` confidence |
| Add retry and rate-limit handling | both pipelines make single synchronous calls with no backoff |
| Decide store vs regenerate | see run-to-run reproducibility above |
| Validate the Batch API | halves Azure's cost for bulk work; still untested |
| Separate the NE-50 holdout | 48 of its 50 specimens are currently inside the training set, so the Azure calibration cannot be validly refit |
