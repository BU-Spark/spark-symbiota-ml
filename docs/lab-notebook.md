# Lab notebook

A running record of experiments on the transcription pipelines: what we asked, how
we tested it, what came back, and what we decided. Newest sections at the bottom of
each part.

This is the *log*. Two companion documents draw from it:

- `docs/azure-confidence-pipeline.md` — how the shipped Azure confidence scoring
  works. Reference, not history.
- `docs/cost-projection.md` — what the services cost and what scaling up would cost.

Each entry follows the same shape: **Question → Method → Result → Decision**. If an
experiment changed shipped code, the Decision says so. Numbers are reproducible with
the commands in the Azure doc's runbook; nothing here needs an API key to re-check
unless the entry says it cost money.

Conventions used throughout:

- **rho** is Spearman correlation of confidence against correctness — how well a
  score ranks right answers above wrong ones. Higher is better; 0 is useless;
  negative means actively misleading.
- **Cost is dollars per 1,000 specimens**, measured from real token usage on this
  corpus, not vendor estimates.
- **The eval set** is `transcription/data/gbif-ne-500`: ~530 New England herbarium
  sheets whose GBIF record supplies both the image and the Darwin Core ground truth.

---

# Part 1 — Azure pipeline (Azure OCR + gpt-4o-mini)

## 1.1 Which confidence signal per field

**Question.** The pipeline extracts six fields. What actually predicts whether a
given field is right?

**Method.** Score each candidate signal on the cached eval set and rank by rho.
Candidates: the LLM's own self-rating, OCR grounding (did the value appear in the
OCR text), and reference checks against external authorities.

**Result.** The LLM's self-rating does not predict correctness and was discarded.
OCR grounding is weak — the common failure is misreading text that *is* on the
label, not inventing text that isn't, so grounding stays high on exactly the errors
we care about. Reference checks won on every field: GBIF taxonomic backbone for
`scientificName`, a place gazetteer for `location`, a known-collector list for
`recordedBy`, digit-run structure for `barcode`, date validity for `eventDate`.

**Decision.** Confidence is built from reference checks. `run_doc_intell_pipeline`
explicitly drops the model's self-rating. `institutionCode` is left unscored — no
trustworthy ground truth, and it is not in the middleware contract.

## 1.2 Do the paid "enhanced" signals earn their cost

**Question.** Self-consistency (re-extract K=3 at temperature 0.8) and an
independent vision read double the pipeline cost, $2.00 → $4.50 per 1,000
specimens. Is that worth it?

**Method.** Score every field twice off the caches — base signals only, then with
self-consistency and vision — and compare rho and missed-error rate.
`nbs/roi_pareto_eval.py`.

**Result.**

| Field | rho base | rho enhanced | miss base | miss enhanced |
|---|---:|---:|---:|---:|
| `scientificName` | +0.449 | +0.483 | 25.4% | 15.5% |
| `eventDate` | +0.144 | +0.363 | 18.2% | 8.8% |
| `recordedBy` | +0.160 | +0.406 | 16.3% | 8.7% |
| `barcode` | +0.188 | +0.409 | 11.9% | 9.8% |

Three of four fields roughly triple their discrimination. `scientificName` barely
moves — GBIF already gives it a strong free signal, so it is carried by the others
rather than carrying them.

**Decision.** `CONFIDENCE_ENHANCED` defaults to **on**. Without it, `eventDate`,
`recordedBy`, and `barcode` are too weak to route review on. One self-consistency
batch and one vision call serve all five fields, so the cost is per specimen, not
per field.

## 1.3 Calibration

**Question.** Raw signals sit on arbitrary scales — `location` emits multiples of a
third, `scientificName` only 0.0 / 0.5 / 1.0. A raw 0.8 did not mean "80% likely
correct". Can we make the number mean something?

**Method.** Fit per-field isotonic regression mapping raw score to observed
probability of correctness, on the GBIF set, validating expected calibration error
on the 50-specimen New England holdout. `nbs/fit_calibration.py`.

**Result.** Calibration cut holdout ECE substantially for `scientificName` (0.306 →
0.082), `eventDate` (0.181 → 0.123), and `barcode` (0.265 → 0.171). It made
`recordedBy` *worse* on holdout (0.140 → 0.182), so that field ships uncalibrated.

**Decision.** `transcription/calibration.json` ships maps for `scientificName`,
`eventDate`, `barcode`, and `location`. Isotonic regression is monotonic, so
calibration never reorders two values — it only makes the numbers mean what they
say. Re-fit after any signal change, or the map points at a scale that no longer
exists.

## 1.4 The `location` signal was inverted by a missing dependency

**Question.** `location` measured rho −0.148 — anti-correlated, meaning a reviewer
trusting it is steered wrong. Why?

**Method.** Bucket predictions by confidence value and inspect what lands in each.

**Result.** `_state_code` was returning `None` for all 493 predictions, including
obvious ones like `Grand Rapids, Michigan`. The cause was environmental:
`geonamescache` was not installed, and both `_geo()` and `_state_index()` caught the
`ImportError` and degraded silently. With no gazetteer the field fell through to OCR
grounding — a terse prediction like `Plainville` is perfectly grounded and scores
1.0 while recalling none of the county or state ground truth.

| `geonamescache` | `location` rho |
|---|---:|
| installed | +0.414 |
| missing | −0.148 |

**Decision.** Added a one-shot `RuntimeWarning` so the degradation is visible rather
than silent. The scores stay inside [0,1] and look plausible either way, so nothing
downstream can detect this on its own.

## 1.5 Redesigning the `location` signal

**Question.** With the gazetteer installed, `location` sat at +0.414 — the weakest
of the five fields. Could the vision cross-check be used better?

**Method.** Two changes tested separately off the cache. First, add traditional
state abbreviations (`Conn.`, `Mass.`, `N.H.`) that geonamescache does not carry, so
more predictions resolve a state. Then compare seven formulas for combining the
graded place count with the vision state check.

**Result.** The abbreviations *hurt* on their own: +0.414 → +0.365. They pushed more
predictions onto the binary state-agreement path (1.0 / 0.2), which collapses the
graded ordering into two large ties, and Spearman rewards fine ranking. That
exposed the real problem — the shipped design finished **last of seven** formulas
tested, behind even ignoring the vision signal entirely:

| formula | rho |
|---|---:|
| 0.75·completeness + 0.25·agreement | +0.544 |
| completeness, halved on state disagreement | +0.534 |
| completeness only, vision ignored | +0.532 |
| mean(agreement, completeness) | +0.528 |
| agreement × completeness | +0.518 |
| binary state agreement (as shipped) | +0.434 |

(Measured on an n=530 harness, so not directly comparable to the eval's n=493; all
seven share identical rows, so the ordering holds.)

Two reads agreeing on the state says nothing about the locality and county, which
carry most of the accuracy metric.

**Decision.** Inverted the relationship: the graded place count is now the score and
a contradicting vision state *halves* it. Chose that over the marginally better
fitted blend to avoid tuning weights on a single dataset, and because it mirrors the
year-disagreement penalty `eventdate_confidence` already uses. Kept the
abbreviations, which are neutral-to-positive under the new shape.

**Outcome: `location` rho +0.414 → +0.472**, second best of the five fields.
Calibration was re-fit afterwards, since the score distribution changed shape.

## 1.6 Confidence performance summary

Per-field discrimination, enhanced signals on, after 1.5:

| Field | rho | auto-accepted at ≥0.90 | wrong among them |
|---|---:|---:|---:|
| `scientificName` | +0.48 | 41% | 15.5% |
| `location` | +0.47 | — | — |
| `barcode` | +0.41 | 90% | 9.8% |
| `recordedBy` | +0.39 | 23% | 10.3% |
| `eventDate` | +0.36 | 35% | 8.8% |

At a 0.90 cutoff this auto-accepts about 47% of field values and lets through errors
on about 5% of all fields. On the 1.5M-specimen New England corpus at 10 seconds of
review per field, that is roughly 7,900 reviewer-hours saved — about 4.5
person-years.

**Open questions.** `location` calibration is fit against state correctness while
the eval scores admin token-recall, so the two answer different questions. And the
evidence is partly circular: ground truth comes from GBIF and `scientificName`'s
strongest signal is a GBIF backbone match. The planned hand-labelled non-GBIF set
addresses the second.

## 1.7 The shipped pipeline invents 10.5% of species names

**Question.** How often does the Azure pipeline emit a scientific name that does not
exist?

**Method.** Look up every non-UNKNOWN `scientificName` from the gpt-4o-mini
extraction against the GBIF taxonomic backbone. A name GBIF cannot match at any rank
is not a real name.

**Result.** 15 of 143 names, **10.5%**, cannot be matched. Roughly one in ten
scientific names the pipeline produces is a binomial that does not exist.

The GBIF correction step in `doc_intelligence.py` does not fix these. It rewrites a
read onto GBIF's accepted species on an EXACT match (a valid name or a synonym) or a
FUZZY match (a close misread); a name GBIF cannot match at all passes through
unchanged. So this rate survives correction and reaches the portal.

For scale, image-native models reading the same 150 specimens invent between 0% and
2.7% (see 2.4). This is a property of the OCR-plus-text-LLM approach rather than of
the task: gpt-4o-mini never sees the sheet, so when Azure OCR garbles a handwritten
name it has nothing to fall back on and completes the fragment into something
plausible-looking.

**Decision.** Two things follow. Invented names are cheaply detectable — GBIF
already runs in the pipeline — so they can be flagged for review instead of shipped
silently. And this is a strong argument for the image-native pipeline independent of
the accuracy comparison: 0% invention against 10.5% is a larger difference than any
accuracy gap measured between models.

---

# Part 2 — Anthropic pipeline

## 2.1 Harness and prompt defects found before measuring

**Question.** Can the Anthropic pipeline (`transcription/claude_sonnet.py`, image
straight to the model with no OCR step) be compared fairly against models and
against the shipped Azure pipeline?

**Method.** Run models over the same specimens, cache each JSON output, and score
with the same lenient correctness metrics the Azure work uses.
`nbs/model_run.py` and `nbs/model_compare.py`.

**Result.** Three defects made the first comparison unsafe to read:

1. **The prompt never interpolated its output format.** The literal string
   `{{OUTPUT_FORMAT}}` reached the model, so JSON key names were never pinned.
   Models substituted equivalent Darwin Core synonyms — Sonnet 5 emitted `locality`
   instead of `location` on 11% of records and `catalogNumber` instead of `barcode`
   on 8%. This is a production bug, not just an eval artifact: `envelope.py` reads
   those keys literally, so a drifting model yields empty fields in the portal.
2. **`max_tokens=1024` truncated replies mid-JSON.** The ceiling caps thinking and
   response text together, so models that think before answering lost whole records
   — 5 of 482 Sonnet 5 outputs.
3. **Thinking asymmetry.** Sonnet 5 and Opus 5 think by default; Haiku 4.5 and the
   4.6-generation models do not. Comparing across that boundary charges some models
   a surcharge others do not pay.

**Decision.** Interpolated the output format and added "Use exactly these JSON
keys"; raised `max_tokens` to 4096 (only generated tokens are billed); dropped a
dangling `<input_image>` placeholder that referenced a variable never substituted.
The scorer also accepts Darwin Core synonyms so a model is graded on transcription
rather than key naming. All caches produced under the old prompt were set aside as
not comparable.

## 2.2 Model comparison — accuracy and cost

**Question.** Which Anthropic model gives the best accuracy per dollar on this task?
Frontier models are plausibly overkill for reading a label, so start at the cheap
end and scale up only if the results demand it.

**Method.** 150 specimens per model, identical prompt, identical specimen set,
scored with the same lenient correctness metrics as the Azure work. All three models
have thinking off by default, so none is charged a reasoning surcharge the others
avoid. `nbs/model_run.py` then `nbs/model_compare.py`. Cost $2.96 total.

The shipped Azure pipeline (Azure OCR → gpt-4o-mini) is scored on the same
specimens as a baseline.

The lineup started at the cheap end (Opus 4.6, Sonnet 4.6, Haiku 4.5) and was
extended upward with Sonnet 5 and Opus 4.8 to test whether the premium tier earns
its price. Total spend for all five: $7.63.

**Result.** Per-field accuracy, mean across the five fields, and measured cost. Both
pipelines apply the GBIF taxon correction described in 2.3, so these are the numbers
each ships:

| Model | scientificName | eventDate | recordedBy | barcode | location | mean | $/1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Opus 4.8 | 88.7% | 88.6% | 89.7% | 92.7% | 89.2% | **89.8%** | $22.20 |
| Opus 4.6 | 88.7% | 81.3% | 87.8% | 93.3% | 95.4% | **89.3%** | $10.90 |
| Sonnet 4.6 | 88.7% | 84.0% | 88.4% | 94.7% | 87.7% | **88.7%** | $6.66 |
| Sonnet 5 | 85.3% | 90.0% | 84.6% | 93.3% | 89.2% | **88.5%** | $9.78 |
| gpt-4o-mini (shipped Azure) | 76.2% | 86.4% | 87.0% | 95.9% | 44.3% | **78.0%** | — |
| Haiku 4.5 | 56.2% | 31.0% | 77.0% | 90.0% | 59.4% | **62.7%** | $2.20 |

No model produced an impossible collection year. Fabrication rates differ sharply
between models and are covered in 2.4.

Zero parse or API errors across all 750 calls, versus 3-of-3 parse failures on a
pre-fix Sonnet 4.6 probe and 24 errors in a pre-fix Opus 5 run.

**Reading the numbers.**

*Cost rises 3.3x while accuracy moves 1.1 points.* From Sonnet 4.6 ($6.66, 88.7%) to
Opus 4.8 ($22.20, 89.8%) is more than triple the money for a gap well inside noise —
one standard error on a per-field proportion near 89% is about 2.6 points at n=150.
The four viable models span 88.5–89.8%. On this task they are, statistically, the
same model at very different prices.

*Newer is not better here.* Sonnet 5 scores below Sonnet 4.6 (88.5% vs 88.7%) while
costing 47% more, and its introductory pricing ends 2026-08-31. Opus 4.8 barely
separates from Opus 4.6 at double the price. Nothing in the current generation earns
its premium for this workload.

*Haiku 4.5 is not viable.* `eventDate` at 31% is the collapse — it cannot read
handwritten dates. It also abstains far more (20 `UNKNOWN` scientific names against
zero for the others), so even 62.7% is computed on an easier subset than it looks.

*Per-field leaders differ*, which matters if one field dominates the use case:
Sonnet 5 leads `eventDate` (90.0%), Opus 4.6 leads `location` (95.4%), Sonnet 4.6
leads `barcode` (94.7%), Opus 4.8 leads `recordedBy` (89.7%). The three 4.6/4.8-era
models tie exactly on `scientificName` — see 2.3.

*Every viable model beats the shipped Azure pipeline by ~11 points*, driven by
`location` (87.7–95.4% vs 44.3%) and `scientificName` (88.7% vs 76.2%). Azure stays
ahead on `barcode` and competitive on `eventDate`, the fields its OCR-grounding and
self-consistency signals serve best.

**Decision.** **Sonnet 4.6 is the pick** at $6.66 per 1,000: statistically
indistinguishable from models costing up to 3.3x more. Opus 4.6 is the upgrade if
`location` specifically matters, and that is the one gap approaching significance —
though it is measured on n≈65. Haiku 4.5 is ruled out. Nothing reached the 92% mean
that would have justified paying for the premium tier.

**Caveats.**

- `location` is scored on n≈65, not 150, because ground-truth locality coverage is
  lower than for the other fields. Every `location` conclusion is on a quarter of
  the sample.
- The cost column is **extraction only**. The Anthropic pipeline has no confidence
  scoring, so it is not yet a like-for-like replacement for the Azure pipeline,
  whose $4.50 per 1,000 buys six calls including the confidence signals. Comparing
  $6.66 against $4.50 understates what an Anthropic pipeline would cost once it
  produces a confidence score.
- Sonnet 5's price is introductory and rises to roughly $14.70 per 1,000 after
  2026-08-31, which weakens its case further.

## 2.3 `scientificName` is capped by taxonomy, not by model quality

**Question.** Opus 4.6, Opus 4.8, and Sonnet 4.6 all scored *exactly* 72.7% on
`scientificName`. Identical accuracy across different models suggests a shared
ceiling rather than coincidence. What is it?

**Method.** Compare the sets of specimens each model gets wrong, and inspect the
cases every model fails.

**Result.** 41% of all `scientificName` errors are made by every model. The shared
failures are not misreads:

| every model read | GBIF ground truth says |
|---|---|
| `Cypripedium pubescens` | `Cypripedium parviflorum` |
| `Lonicera parviflora` | `Lonicera dioica` |

These are **taxonomic synonyms**. The models transcribed the label faithfully; GBIF
supplies the *currently accepted* name after revision, so a correct reading of a
historic label scores wrong. No model can beat this, because it is not a
transcription problem.

`GbifTaxonMatcher.correct()` resolves this: it snaps a read onto GBIF's accepted
species on an EXACT match (a valid name or a synonym) or a FUZZY match (a light
misread), and leaves a name GBIF cannot match alone. The original read is preserved
in `verbatimScientificName`, so nothing is lost.

| Model | raw read | corrected | gain | mean before → after |
|---|---:|---:|---:|---|
| Opus 4.8 | 72.7% | 88.7% | +16.0 | 86.6% → **89.8%** |
| Opus 4.6 | 72.7% | 88.7% | +16.0 | 86.1% → **89.3%** |
| Sonnet 4.6 | 72.7% | 88.7% | +16.0 | 85.5% → **88.7%** |
| Sonnet 5 | 72.0% | 85.3% | +13.3 | 85.8% → **88.5%** |
| Haiku 4.5 | 46.9% | 56.2% | +9.2 | 60.9% → **62.7%** |
| gpt-4o-mini (Azure) | 61.5% | 76.2% | +14.7 | 75.0% → **78.0%** |

**Decision.** A free correction step is worth more than any amount of model spend:
+16 points on `scientificName` and +3.2 on the mean for Sonnet 4.6, against +1.1
points for paying 3.3x more for Opus 4.8. `transcription/doc_intelligence.py`
already applied it to the Azure output; it is now also applied by
`claude_sonnet.apply_taxon_correction()` inside `run_claude_pipeline`, so both
pipelines correct. The correction is idempotent, and `nbs/model_compare.py --raw`
scores the model's unmodified read for comparison.

Note the three 4.6/4.8-generation models converge on exactly 88.7% after correction
as well — a second ceiling, this time from labels GBIF cannot resolve at all.

**Best result achieved: 89.8%** (Opus 4.8). Nothing reached 92%, and the gap between
the cheapest viable model and the most expensive stays close to noise.

## 2.4 Hallucination rate

**Question.** Accuracy says whether an answer is wrong. It does not say whether the
model misread the label or invented something. Which models fabricate?

**Method.** Measured against GBIF and the ground-truth labels only — the Azure OCR
is not used, so this is a property of the Anthropic pipeline alone. Every non-UNKNOWN
answer falls into one of four buckets:

- **correct** — matches ground truth.
- **near miss** — the model was reading the right thing and got characters wrong:
  right genus with the wrong species, a barcode within two digits, part of a
  collector's name. A transcription error.
- **far miss** — no relation to the truth.
- **invented** — `scientificName` only: a binomial GBIF's taxonomic backbone cannot
  match at any rank. Unambiguous fabrication, since the name does not exist.

**Result.** `scientificName`, the field where fabrication is detectable. Scored on
the model's **raw read**, before the taxon correction in 2.3, since the question is
what the model itself produced. Correction does not change the invented column — a
name GBIF cannot match is exactly the name correction cannot fix.

| Model | correct | near miss | far miss | **invented** |
|---|---:|---:|---:|---:|
| Opus 4.8 | 72.7% | 15.3% | 12.0% | **0.0%** |
| Sonnet 4.6 | 72.7% | 15.3% | 12.0% | **0.0%** |
| Opus 4.6 | 72.7% | 14.7% | 12.0% | **0.7%** |
| Sonnet 5 | 72.0% | 14.7% | 12.7% | **2.7%** |
| Haiku 4.5 | 46.9% | 20.0% | 24.6% | **10.8%** |

Examples of invented names, all from Haiku 4.5:

| model read | ground truth |
|---|---|
| `Thyrica aethiifolia` | `Comptonia peregrina` |
| `Impala Glochoma` | `Glechoma hederacea` |
| `Antennaria felixGreene` | `Antennaria parlinii` |

`barcode` and `recordedBy` show no invented values for any model. Barcode errors are
overwhelmingly near misses — a digit or two misread (3.3–8.7%) with far misses at
0–3.3%. `recordedBy` errors are almost entirely far misses (10.3–15.4%, and 23.0%
for Haiku 4.5), meaning a wrong collector is usually a different person rather than
a misspelling. No model produced an impossible collection year.

**Reading the numbers.** About 15% of every good model's `scientificName` answers are
near misses — right genus, wrong species. That is transcription difficulty on
handwritten cursive, not invention, and it is the dominant error mode.

Fabrication separates the models where accuracy does not. Opus 4.8 and Sonnet 4.6
invent nothing. Sonnet 5 invents 2.7% while the cheaper Sonnet 4.6 invents 0%,
adding to the case against it. Haiku 4.5 at 10.8% is producing names that do not
exist roughly once every nine specimens.

**Decision.** Fabrication is the third independent reason to rule out Haiku 4.5,
after its 31% `eventDate` accuracy and its 62.7% mean. Among the viable models it
favours Sonnet 4.6 and Opus 4.8, both at zero. Since invention is detectable — a
name GBIF cannot match is fabricated by definition — it can also be caught
automatically and routed to review rather than shipped.

## 2.5 Cross-model agreement as a confidence signal

**Question.** The Anthropic pipeline ships values with no indication of
reliability. The Azure pipeline's strongest signals are all reference checks —
compare an answer against an independent second read. Can a second *model* play
that role?

**Method.** Two models read the same specimen. One is the **primary**, whose answer
would ship; the other is the **checker**, used only to decide whether to trust the
primary. For each ordered pair and field, measure the primary's accuracy where the
checker agrees against where it does not. Agreement uses the same lenient
comparators as the accuracy scoring, on the raw reads — the GBIF correction snaps
different synonyms onto one accepted name and would inflate agreement for free.
150 specimens, all five models, `nbs/model_agreement.py`. No API cost.

**Result.** With Sonnet 4.6 as the primary, by checker:

| Field | Opus 4.6 | Opus 4.8 | Sonnet 5 | Haiku 4.5 | Azure signal |
|---|---:|---:|---:|---:|---:|
| `scientificName` | **+0.352** | +0.157 | +0.303 | +0.125 | +0.483 |
| `eventDate` | +0.633 | **+0.885** | +0.817 | +0.151 | +0.363 |
| `recordedBy` | **+0.676** | +0.676 | +0.668 | +0.323 | +0.394 |
| `barcode` | **+0.510** | +0.254 | +0.369 | +0.415 | +0.409 |
| `location` | — | — | — | +0.184 | +0.472 |

The strongest cell, Sonnet 4.6 checked by Opus 4.8 on `eventDate`: the two agree on
88% of specimens, and where they agree Sonnet 4.6 is right **96.8%** of the time;
where they disagree it is right **0.0%**. For comparison, the Azure pipeline's
`eventDate` confidence at its 0.90 cutoff auto-accepts 35% of values with 8.8% of
those wrong.

**Reading the numbers.** The pattern tracks whether an external authority exists.
GBIF can adjudicate a species name and a gazetteer can adjudicate a place, so a
reference check wins on `scientificName` and `location`. Nothing can adjudicate a
handwritten date, so a second opinion is the only available signal there — and it is
much stronger than anything the Azure pipeline achieves on that field.

Haiku 4.5 is the weakest checker on three of five fields. A checker does not need to
be accurate, only independently wrong, but Haiku is wrong too often to be
informative.

**The cost is the constraint.** Every checker re-reads the whole image, and image
ingestion dominates the cost of this pipeline:

| Configuration | $/1,000 |
|---|---:|
| Sonnet 4.6 alone | $6.66 |
| + Haiku 4.5 checker | $8.86 |
| + Sonnet 5 checker | $16.44 |
| + Opus 4.6 checker | $17.56 |
| + Opus 4.8 checker | $28.86 |

The entire Azure pipeline including its confidence signals costs $4.50 per 1,000.
It gets self-consistency at K=3 plus a vision read for +$2.50 because its expensive
step, the OCR, happens once and each additional read is a text-only call. An
image-native pipeline has no cheap second opinion available.

**Decision.** Cross-model agreement works, and on `eventDate` it is the best signal
measured anywhere in this project. It is not adopted as a default because it costs
164% more than the extraction it protects. Deferred pending two cheaper routes:
checking selectively rather than on every specimen, and cross-pipeline reuse where
Azure OCR has already been paid for. The free signals in 2.6 are adopted instead.

**Caveat.** The good models agree 93–95% on `scientificName` and `location`, leaving
8–10 disagreements at n=150, so those cells are directional. `eventDate` rests on
about 18 disagreements.

## 2.6 A free confidence layer for the Anthropic pipeline

**Question.** Which fields can be scored with no OCR text and no second model call?

**Method.** Two references cost nothing to consult and are already in the codebase:
the GBIF taxonomic backbone, which the pipeline already queries for the taxon
correction, and the offline place gazetteer used by the Azure `location` signal.
Score on the raw model read, before the correction, and measure against ground
truth on 530 specimens.

**Result.**

| Field | free signal | rho | Azure (paid) | accuracy by band |
|---|---|---:|---:|---|
| `scientificName` | GBIF match type | **+0.418** | +0.483 | high 91% · low **14%** (n=29) |
| `location` | gazetteer place count | **+0.428** | +0.472 | high 79% · mid 65% · low 42% |
| `recordedBy` | collector gazetteer | +0.079 | +0.394 | high 91% · mid 87% |

`scientificName` and `location` reach roughly 87–91% of the discrimination the Azure
pipeline gets from paid signals, at no cost. The `scientificName` low band is the
most useful cell in the table: 29 specimens where the answer is 86% likely wrong,
identified for free.

`recordedBy` does not survive. Its bands differ by 4 points and only 2 specimens fall
below 0.4. `eventDate` and `barcode` have no free signal at all — their Azure
fallbacks both need OCR words and collapse to a constant without them.

**Two ordering constraints found while wiring this.** Confidence must be computed on
the raw read **before** the taxon correction; correcting first makes every name match
GBIF and saturates the signal (rho falls from +0.418 to +0.270). And the JSON
extractor must tolerate fenced and preamble-wrapped responses, or the correction and
scoring silently skip every model that wraps its output.

**Decision.** `claude_sonnet.build_free_confidence()` ships `scientificName` and
`location` only. Fields with no usable signal are omitted rather than scored weakly —
`envelope.py` renders a missing field as "confidence unavailable" and shows no score
bar, whereas a low-information score would be displayed as though it meant something.
Scores are uncalibrated: `transcription/calibration.json` was fit on the Azure
pipeline's score distributions and does not transfer.

This closes part of the gap that kept the Anthropic pipeline from being a drop-in
replacement. `eventDate`, `recordedBy`, and `barcode` remain unscored.
