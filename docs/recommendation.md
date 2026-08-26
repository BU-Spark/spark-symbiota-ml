# Pipeline recommendation

What to ship, what it costs, and when to use which. Evidence is in
`docs/lab-notebook.md`; prices are in `docs/cost-projection.md`.

## Recommendation

**Offer all three. Default to Google for bulk digitisation, and offer Claude Sonnet 5
for collections where the record will be published or relied on.**

| | Azure | Google | Anthropic |
|---|---|---|---|
| Extractor | Azure OCR + gpt-4o-mini | Google Doc AI + gpt-4o-mini | Claude Sonnet 5, image-native |
| Accuracy | 74% | 76% | **85%** |
| Invented species names | 12.3% | 6.0% | **0.8%** |
| Cost per 1,000, with confidence | $4.50 | **$3.00** | $19.90 |
| Confidence | 5 fields, all calibrated | 5 fields, all calibrated | 5 fields, all calibrated |

**Google replaces Azure as the cheap default.** It is more accurate at two thirds the
price, and invents half as many species names. Both use the same language model;
only the OCR engine and prompt differ.

Claude costs ten times Google and is 9 accuracy points better. That is a real tradeoff,
not an upgrade, which is why all three stay available.

**If only one can be maintained, ship Claude Sonnet 5.** Two of the Azure gaps are
severe enough to affect published data: it reads `location` correctly 55% of the
time, and roughly one in eight scientific names it emits does not exist.

## The options, priced

| Option | Accuracy | $/1,000 | Confidence |
|---|---:|---:|---|
| **Google, full** | **76%** | **$3.00** | 5 fields |
| Google, extraction only | 76% | $2.00 | none |
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

Per-field accuracy, on the same 150 specimens:

| Field | Azure | Google | Claude Sonnet 5 |
|---|---:|---:|---:|
| `location` | 54.5% | 65.7% | **85.6%** |
| `scientificName` | 71.4% | 76.9% | **85.3%** |
| `eventDate` | 78.9% | 76.7% | **87.2%** |
| `barcode` | 90.5% | 89.1% | **91.3%** |
| `recordedBy` | **75.3%** | 71.5% | 75.0% |

The gap is concentrated in `location` and `scientificName`, and both have the same
cause: gpt-4o-mini never sees the specimen. It only receives whatever text Azure's OCR
managed to extract, so when the OCR garbles handwritten cursive it has nothing to
check against and completes the fragment into something plausible. A model looking at
the sheet can read the handwriting.

That also explains the fabrication rate. Names that do not exist in the global species
database:

| Pipeline | Invented species names |
|---|---:|
| Azure (gpt-4o-mini) | 12.3% |
| Google (gpt-4o-mini) | 6.0% |
| Claude Sonnet 5 | **0.8%** |

An earlier version of this document reported 11.3% and 1.3%. That measurement counted
a capitalised species epithet (*Rumex Acetosella*, a 19th-century labelling
convention) as invented, which penalised the pipeline that transcribes most
faithfully. Corrected above.

These are automatically detectable — a name GBIF cannot match at any rank is
fabricated by definition — so they can be flagged for review on either pipeline rather
than shipped silently. That is worth doing regardless of which pipeline is chosen.

## What each pipeline ships with

All three produce the same flat Darwin Core object plus a parallel `_confidence` map,
so they are interchangeable behind a selector.

| Field | Azure signal | Google signal | Claude signal |
|---|---|---|---|
| `scientificName` | GBIF match + vision | GBIF match + vision | GBIF match |
| `location` | gazetteer + vision state check | gazetteer + vision | gazetteer |
| `eventDate` | self-consistency + vision | vision agreement | checker model |
| `recordedBy` | collector list + vision | collector list + vision | collector list + checker |
| `barcode` | digit-run + self-consistency | digit-run structure | agreement with the Azure read |

All three carry calibrated maps on all five fields, so a 0.9 means the same thing
whichever pipeline produced it. The maps are deliberately conservative: each value is
the figure we are 90% confident the data supports, not the observed average. They
under-state rather than over-state, verified against 148 specimens never used for
fitting. See [confidence.md](confidence.md).

## Setting the review threshold

There is no single right cutoff, and it is a per-collection decision rather than a
project-wide one. **0.80 to 0.85 is the useful range** — at 0.90 only two of Azure's
five fields can reach the cutoff at all, because the maps under-state deliberately.

What each cutoff buys, per field and per pipeline, is in
`transcription/threshold_table.json`. The tool at `/tool` shows it live.

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
source. A 40-specimen hand-labelled check found **no pipeline errors** among the
`scientificName` disagreements — every one was a transcription slip, an abbreviation,
or a synonym in the hand labels. So no inflation was detected, but the check is too
small and too noisy to confirm the numbers: a non-expert labels cursive species names
at about 11% error, which cannot validate a pipeline claiming 85%. Closing this needs
a botanist on the species column.

**The model comparison is n=150.** Differences under about 3 points are inside noise.
The 10-point gaps are safe; the 0.4-point gap between Sonnet 5 and Opus 4.8 is not
meaningful.

**No pipeline is reproducible run to run.** Five runs of the same specimen change
at least one field on 40% of specimens — usually formatting or completeness, sometimes
a genuine disagreement. Aggregating five runs does not improve accuracy, so the answer
is to store results rather than regenerate them on demand.

**`location` ground truth covers about two-thirds of specimens**, so that column rests
on a smaller sample than the others.

## What to do next

**Get a botanist to review 40 species names.** Roughly twenty minutes of expert time.
It is the only thing standing between "no inflation detected" and "the numbers are
confirmed", and every accuracy figure here carries that caveat until it is done.

**Let the tool learn from corrections.** Confidence means different things in
different collections — the same field reads 39% correct on amateur handwritten
labels and 75% on printed ones. The maps are conservative so this is safe rather than
solved, but the real fix is fitting per collection. Every correction typed into
`/tool` is a labelled specimen from that collection; nothing reads that log yet, and
roughly 150 to 300 reviews would be needed before a refit means anything.

**Harden for scale.** No retry, no backoff, no parallelism. Fine for 150 specimens; a
1.5M run would take a year and lose work to transient failures.

**Decide store versus regenerate.** Five runs of the same specimen change at least one
field on 40% of specimens, and averaging runs does not help. Results should be stored,
not recomputed on demand.

**Redraw the holdout.** The current one was drawn from the largest herbaria, so it is
easier than the training data. It proves nothing over-states, but understates how much
review the tool can save.

**Benchmark the open-source alternatives.** The summer plan lists them and none has
been tried. The free Tesseract baseline has still never been scored, so there is no
floor to compare the paid services against.
