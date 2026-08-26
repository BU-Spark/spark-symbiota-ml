# Lab notebook

A running record of experiments on the transcription pipelines: what we asked, how
we tested it, what came back, and what we decided. Newest sections at the bottom of
each part.

This is the *log*. Two companion documents draw from it:

- `docs/confidence.md` — how the shipped confidence scoring
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

> **Superseded by 4.2.** These figures predate conservative calibration. The maps now
> under-state deliberately, so a 0.90 cutoff covers far fewer fields. Current
> coverage and precision at every cutoff: `transcription/threshold_table.json`.
> The `location` open question below is resolved — it is now fitted against the
> displayed locality string.

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

## 1.7 The shipped pipeline invents 11% of species names

**Question.** How often does the Azure pipeline emit a scientific name that does not
exist?

**Method.** Look up every non-UNKNOWN `scientificName` from the gpt-4o-mini
extraction against the GBIF taxonomic backbone. A name GBIF cannot match at any rank
is not a real name.

> **Superseded by 4.6.** This count included real species with a capitalised
> epithet, which GBIF does not match. The corrected figure is 12.3%.

**Result.** 15 of 133 names, **11.3%**, cannot be matched. Roughly one in nine
scientific names the pipeline produces is a binomial that does not exist.

The GBIF correction step in `doc_intelligence.py` does not fix these. It rewrites a
read onto GBIF's accepted species on an EXACT match (a valid name or a synonym) or a
FUZZY match (a close misread); a name GBIF cannot match at all passes through
unchanged. So this rate survives correction and reaches the portal.

For scale, image-native models reading the same specimens invent 0.7–1.3% (see 2.4).
This is a property of the OCR-plus-text-LLM approach rather than of the task:
gpt-4o-mini never sees the sheet, so when Azure OCR garbles a handwritten name it has
nothing to fall back on and completes the fragment into something plausible-looking.

**Decision.** Two things follow. Invented names are cheaply detectable — GBIF already
runs in the pipeline — so they can be flagged for review instead of shipped silently.
And this is an argument for the image-native pipeline independent of the accuracy
comparison: roughly 1% invention against 11% is a larger difference than any accuracy
gap measured between models.

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

**A sampling bug invalidated the first attempt.** Specimens were selected by taking
the first N in filename order. GBIF assigns occurrence keys in batches per published
dataset, so filename order clusters by institution and the first 150 turned out to be
149 sheets from one museum, all sharing one catalog-number format. Barcode accuracy
read 96% on that set and 56% across the corpus. `model_run.py` now samples with a
seed, and every number below is from the re-run.

**Result.** Per-field accuracy, mean across the five fields, and measured cost. Both
pipelines apply the GBIF taxon correction from 2.3, so these are the numbers each
ships:

| Model | scientificName | eventDate | recordedBy | barcode | location | mean | $/1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Opus 4.8 | 84.7% | 87.2% | 83.3% | 90.0% | 81.6% | **85.3%** | $25.81 |
| Sonnet 5 | 85.3% | 87.2% | 75.0% | 91.3% | 85.6% | **84.9%** | $11.34 |
| Opus 4.6 | 85.3% | 81.8% | 81.8% | 56.6% | 87.7% | **78.6%** | $10.78 |
| Sonnet 4.6 | 80.0% | 83.0% | 79.2% | 53.7% | 79.6% | **75.1%** | $6.56 |
| gpt-4o-mini (shipped Azure) | 71.4% | 78.9% | 75.3% | 90.5% | 54.5% | **74.1%** | $4.50 |
| Haiku 4.5 | 54.9% | 39.1% | 66.4% | 50.9% | 34.0% | **49.1%** | $2.17 |

No model produced an impossible collection year. Fabrication rates are in 2.4. Zero
parse or API errors across 750 calls, against 3-of-3 parse failures on a pre-fix
Sonnet 4.6 probe and 24 errors in a pre-fix Opus 5 run.

**Reading the numbers.**

*`barcode` splits the field by generation.* Opus 4.8 and Sonnet 5 read 90%; every
4.6-era model reads about 55%. A 35-point gap, and it is the single biggest driver of
the ranking. The uniform Yale set hid it entirely, because one institution means one
format. The wider corpus is 470 prefixed numbers (`YU.037310`), 58 bare digits, and 2
other — and a bare number is easy to confuse with the accession numbers, elevations
and plot IDs also printed on a sheet.

*Sonnet 5 is the value pick* at $11.34 per 1,000. It is level with Opus 4.8's 85.3%
at less than half the price; the 0.4-point gap is well inside noise, where one
standard error on a per-field proportion near 85% is about 2.9 points at n=150.

*The cheaper tier is not close.* Opus 4.6 and Sonnet 4.6 sit 6 to 10 points behind,
so the earlier conclusion that price buys nothing was an artifact of the biased
sample. Opus 4.6 in particular is now hard to justify: worse than Sonnet 5 at
roughly the same cost.

*Haiku 4.5 is not viable.* `eventDate` at 39.1% is the collapse — it cannot read
handwritten dates — and it abstains far more than the others, so even 49.1% is
computed on an easier subset than it looks.

*Every viable model beats the shipped Azure pipeline*, by 11 points for Sonnet 5,
driven almost entirely by `location` (85.6% vs 54.5%) and `scientificName` (85.3% vs
71.4%). Azure holds its own on `barcode` and `eventDate`, the fields its OCR grounding
and self-consistency serve best.

**Decision.** **Sonnet 5**, at $11.34 per 1,000 for extraction. Opus 4.8 costs 2.3x
more for 0.4 points. Nothing reached the 92% mean that would have justified the
premium tier.

**Caveats.**

- `location` ground truth covers only about two-thirds of specimens, so its column
  rests on a smaller sample than the others.
- The cost column is extraction only. Confidence scoring is priced separately in 2.5
  and 2.6; the Azure figure already includes its own.
- Sonnet 5's price is introductory and rises to roughly $17.51 per 1,000 after
  2026-08-31.

## 2.3 `scientificName` is capped by taxonomy, not by model quality

**Question.** `scientificName` is the weakest field for every model, and the GBIF
correction lifts every one of them by almost exactly the same 16 points. A gain that
uniform across models of very different strength points at a shared ceiling rather
than at model quality. What is it?

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
| Sonnet 5 | 69.3% | 85.3% | +16.0 | 81.7% → **84.9%** |
| Opus 4.6 | 69.3% | 85.3% | +16.0 | 75.4% → **78.6%** |
| Opus 4.8 | 68.7% | 84.7% | +16.0 | 82.1% → **85.3%** |
| Sonnet 4.6 | 64.0% | 80.0% | +16.0 | 71.9% → **75.1%** |
| gpt-4o-mini (Azure) | 58.6% | 71.4% | +12.8 | 71.6% → **74.1%** |
| Haiku 4.5 | 46.6% | 54.9% | +8.3 | 47.4% → **49.1%** |

**Decision.** A free correction step is worth more than any amount of model spend:
+16 points on `scientificName` and +3.2 on the mean, against +0.4 points for paying
2.3x more for Opus 4.8. `transcription/doc_intelligence.py` already applied it to the
Azure output; `claude_sonnet.apply_taxon_correction()` now applies it inside
`run_claude_pipeline`, so both pipelines correct. It is idempotent, and
`nbs/model_compare.py --raw` scores the model's unmodified read for comparison.

Every model gains almost exactly +16 points, which is what a shared ceiling looks
like: the correction is resolving the same set of renamed species regardless of which
model read the label.

**Best result: 85.3%** (Opus 4.8). Nothing reached 92%.

## 2.4 Hallucination rate

> **The invented column is superseded by 4.6.** It counted a capitalised species
> epithet as fabricated. Corrected: Sonnet 5 is 0.8%, not 1.3%.

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
| Sonnet 5 | 85.3% | 11.3% | 2.0% | **1.3%** |
| Opus 4.6 | 85.3% | 11.3% | 2.7% | **0.7%** |
| Opus 4.8 | 84.7% | 11.3% | 3.3% | **0.7%** |
| Sonnet 4.6 | 64.0% | 23.3% | 11.3% | **1.3%** |
| Haiku 4.5 | 54.9% | 17.3% | 20.3% | **7.5%** |
| gpt-4o-mini (shipped Azure) | — | — | — | **11.3%** |

Examples of invented names, all from Haiku 4.5:

| model read | ground truth |
|---|---|
| `Thyrica aethiifolia` | `Comptonia peregrina` |
| `Impala Glochoma` | `Glechoma hederacea` |
| `Antennaria felixGreene` | `Antennaria parlinii` |

No model invented a barcode or a collector, and none produced an impossible year.
`barcode` errors are misreads and misidentified numbers; `recordedBy` errors are
mostly a different person rather than a misspelling.

**Reading the numbers.** About 11% of the good models' `scientificName` answers are
near misses — right genus, wrong species. That is transcription difficulty on
handwritten cursive, not invention, and it is the dominant error mode throughout.

The three strong models invent 0.7–1.3%. Haiku 4.5 invents 7.5%, roughly one name in
thirteen that does not exist.

**The shipped Azure pipeline invents 11.3%** — worse than every Claude model tested,
including Haiku. Section 1.7 covers why: gpt-4o-mini never sees the sheet, so when
Azure's OCR garbles a handwritten name it completes the fragment into something
plausible with nothing to check against.

**Decision.** Fabrication is the third independent reason to rule out Haiku 4.5,
after its 39.1% `eventDate` and its 49.1% mean. Among the viable models the
differences are small enough not to drive the choice. Since invention is detectable —
a name GBIF cannot match is fabricated by definition — it can be flagged
automatically rather than shipped, on either pipeline.

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

**Adopted.** Re-measured on the representative sample with Sonnet 5 as the shipped
model, the checker choice matters less than expected:

| checker | `eventDate` | `recordedBy` | checker cost / 1,000 |
|---|---:|---:|---:|
| Opus 4.8 | +0.624 | — | $25.81 |
| **Sonnet 4.6** | **+0.620** | **+0.406** | **$6.56** |
| Opus 4.6 | +0.500 | +0.508 | $10.78 |
| Haiku 4.5 | +0.250 | +0.259 | $2.17 |

Sonnet 4.6 matches Opus 4.8 on `eventDate` at a quarter of the price, and both beat
the Azure pipeline's own signals for those fields (+0.363 and +0.394).

`claude_sonnet.run_claude_pipeline(..., checker_model=...)` runs the second read and
scores `eventDate` and `recordedBy` from it. It is off by default: the checker
doubles extraction cost, from $11.34 to $17.90 per 1,000 with Sonnet 4.6.

**Caveat.** The good models agree 82–93% on these fields, so each cell rests on
10–25 disagreements at n=150. Directional, not tight.

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

With the two optional reads below, the pipeline scores **all five fields**.

| field | signal | extra call |
|---|---|---|
| `scientificName` | GBIF match type | none |
| `location` | place gazetteer | none |
| `eventDate` | agreement with a checker model | Claude checker |
| `recordedBy` | collector gazetteer + checker agreement | Claude checker |
| `barcode` | agreement with the Azure pipeline's read | Azure OCR + 1 gpt-4o-mini |

`barcode` was the last gap, and only one source works. Measured against Sonnet 5's
reads:

| source | rho | marginal cost / 1,000 |
|---|---:|---:|
| **Azure OCR + one gpt-4o-mini call** | **+0.546** | **$2.00** |
| Opus 4.8 as a Claude checker | +0.393 | $25.81 |
| Sonnet 4.6 checker (already paid for) | +0.158 | $0 |
| Azure OCR words alone | +0.001 | $1.50 |

The Azure route is both cheapest and strongest, and it beats the Azure pipeline's
own barcode signal (+0.409). Two things about it are easy to get wrong:

*The OCR words contribute nothing.* `barcode_confidence` falls back to a
longest-digit-run heuristic when given words but no second read, and that scores
+0.001 for Sonnet 5 and −0.206 for Sonnet 4.6 — worse than useless. The signal is
agreement with gpt-4o-mini's extraction, so the OpenAI call is the part that matters,
not the OCR.

*One call is enough.* Azure's five self-consistency samples score no better than its
single read (+0.465 vs +0.546 for Sonnet 5), so there is no reason to pay for five.

*The signal is weaker for a better model.* Sonnet 4.6 scores +0.802 here against
Sonnet 5's +0.546, because Sonnet 4.6 reads barcodes correctly only 53.7% of the time
and there is more wrongness to detect. The same inversion shows up in cross-model
agreement: a weak primary makes any checker look strong.

**Full confidence stack**, Sonnet 5 as the shipped model:

| component | $/1,000 |
|---|---:|
| Sonnet 5 extraction | $11.34 |
| Sonnet 4.6 checker (`eventDate`, `recordedBy`) | $6.56 |
| Azure OCR + 1 gpt-4o-mini (`barcode`) | $2.00 |
| **all five fields scored** | **$19.90** |


---

# Part 3 — Google pipeline (Google Document AI + gpt-4o-mini)

## 3.1 Bringing the pipeline back and giving it the taxon correction

**Question.** The Google pipeline had not run since the cloud project it used was
lost. Does it still work, and is it comparable with the other two?

**Method.** New Google Cloud project, Document AI processor, and service-account
credentials. Then a diff against `doc_intelligence.py` to find what the two OCR
pipelines did differently.

**Result.** One difference mattered: `google_vision.py` never applied the GBIF taxon
correction. It imported `build_confidence` but not `GbifTaxonMatcher`, so it shipped
whatever binomial the model produced, including authorities and synonyms. Every other
pipeline snaps the name to GBIF's accepted species and keeps the original in
`verbatimScientificName`.

Ported and verified on five specimens: four names were rewritten (authorities
stripped, `Osmunda regalis var. spectabilis` became `Osmunda spectabilis`), and the
fifth, a misread `Cuphrasia kortkomiana`, was correctly left alone and flagged `NONE`.

One difference was left in place: Azure's prompt asks for a structured location object
and is given per-token OCR confidences; Google's uses few-shot examples and asks for a
flat string. Changing it would make cached runs incomparable.

**Decision.** Correction ported. The prompt difference is recorded as a confound: the
Azure/Google comparison below is "Google's OCR and prompt" against "Azure's OCR and
prompt", not an isolated OCR comparison.


## 3.2 Accuracy against the other two pipelines

**Question.** Where does Google land?

**Method.** `nbs/model_compare.py` on the 150 specimens common to all three caches,
same ground truth, GBIF correction applied to every pipeline.

**Result.**

| Field | Azure | Google | Anthropic |
|---|---:|---:|---:|
| `scientificName` | 71.4% | 76.9% | **85.3%** |
| `eventDate` | 78.9% | 76.7% | **87.2%** |
| `recordedBy` | **75.3%** | 71.5% | 75.0% |
| `barcode` | 90.5% | 89.1% | **91.3%** |
| `location` | 54.5% | 65.7% | **85.6%** |
| **mean** | 74.1% | **76.0%** | **84.9%** |
| **$/1,000 (extraction)** | $2.00 | $2.00 | $11.34 |

**Google beats the shipped production pipeline at the same extraction cost**, and at
less than half the cost once confidence is included ($3.00 against $4.50). Both use
the same language model. The gap is concentrated in `location` (+11.2) and
`scientificName` (+5.5) — the two fields where an OCR miss leaves the model completing
a fragment with no way to check it.

**Decision.** Google becomes the recommended default for bulk work, replacing Azure.


## 3.3 Building its confidence layer

**Question.** Google shipped with two signal sources against Azure's four. What is
worth adding?

**Method.** The vision cross-read is a gpt-4o-mini read of the *image*, so it does not
depend on which OCR engine produced the text — the existing cache could be scored
against Google for free. Self-consistency needed a new cache (about $0.25 on 150).

**Result.** Mean AUC, where 0.50 is a coin flip:

| | grounding + GBIF | + vision | + self-consistency |
|---|---:|---:|---:|
| mean AUC | 0.641 | **0.730** | 0.747 |

Vision bought +0.089 for about $1.00 per 1,000. Self-consistency added +0.017 for
$1.50 to $2.50, almost all of it `barcode` (0.646 to 0.725).

**Decision.** Vision shipped, self-consistency not. The gain does not justify doubling
the confidence cost, and `barcode` calibrates without it — its isotonic fit collapses
to a single level *with* self-consistency and holds without it. Google ships three
signal sources at $3.00 per 1,000.


---

# Part 4 — Calibration policy

## 4.1 Confidence means different things in different collections

**Question.** All calibration is fitted on `gbif-ne-500`. Does a map fitted there mean
the same thing elsewhere?

**Method.** Scored the Azure `location` signal on 148 New England specimens drawn from
a different institution mix — the twelve largest herbaria, against a training set that
is half Yale and the New England Botanical Club.

**Result.** The same signal, the same metric, different specimens:

| | accuracy |
|---|---:|
| Training mix (amateur-heavy, handwritten labels) | 39% |
| Holdout mix (large herbaria, printed labels) | 75% |

Nearly double, purely from provenance. A map fitted on one and applied to the other is
wrong by up to 56 points.

**Decision.** This is the finding that drives 4.2. A single global map cannot be
correct for both, so it must at least be wrong in the safe direction.


## 4.2 Making the maps conservative

**Question.** Given 4.1, which way should a map err?

**Method.** Under-confidence wastes reviewer time. Over-confidence tells someone a
field is safe when it is not, and the error enters the database unseen. Only the
second is harmful, so the maps should never over-state.

Each isotonic level now ships the lower end of a one-sided 90% interval on the
observed rate rather than the rate itself. Blocks with few observations shrink most,
which is where over-confidence comes from: 7 observations at 100% ship as 0.81, while
130 at 92% barely move.

**Result.** Two changes to the fitting machinery were required.

Standard calibration error penalises under-confidence exactly as much as
over-confidence, so it rejected every conservative map as worse. Replaced with a
one-sided version counting only over-statement: an under-confident map scores 0.00, an
over-confident one still scores in full.

The lower bound is applied per block *after* PAVA, and it shrinks small blocks hardest
— so a sparse high block can fall below a dense low one, producing a map where a
higher raw score maps to a lower probability. Both fitters now re-impose a running
floor. All 15 field/pipeline maps are monotonic.

**Decision.** Conservative fitting is the default; `--point-estimate` opts out. The
cost is coverage: at a 0.90 cutoff most fields now go to a human. 0.80 to 0.85 is the
useful range, and the tool exposes a slider so a collection can choose.


## 4.3 Validating on specimens never fitted on

**Question.** Every calibration figure so far comes from repeated splits of the
training set, and `min_count` was chosen by looking at those splits. Do the maps hold
on data nobody tuned against?

**Method.** 148 specimens drawn from the 8M-row GBIF export, disjoint from
`gbif-ne-500`, `hand-50`, NE-50 and `raw-images`. Scored each pipeline exactly as it
ships. `nbs/validate_on_holdout.py`.

**Result.** **Nothing over-states, on any field, on any pipeline.** Among values above
a 0.90 cutoff, claimed and delivered agree within 5 points everywhere.

The high `location` errors on this set — 0.33 Azure, 0.24 Google — are
*under*-statement: the maps claim far less than they deliver on this specimen mix.
Safe, but wasteful.

**Decision.** The conservative guarantee holds. One caveat recorded: this set was
drawn with `--institutions 12`, which selects the largest herbaria and so is easier
than the training data. It verifies that nothing over-states — that answer holds
regardless of difficulty — but it understates how much review the tool can save. A
redraw matching the training mix is pending. **Never fit on it**; a holdout is only
useful while untouched.


## 4.4 Behaviour on specimens unlike anything in the eval set

**Question.** A user can upload anything. What happens on sheets from outside New
England?

**Method.** 51 globally distributed specimens — Russia, France, Brazil, Indonesia —
with zero occid overlap with the eval set. Ground truth covers `scientificName` and
`recordedBy` only.

**Result.**

| | in-domain | out-of-domain |
|---|---:|---:|
| `scientificName`, Azure / Google / Anthropic | 71 / 77 / 85% | 73 / 87 / 87% |
| `recordedBy`, Azure / Google | 76 / 72% | 55 / 48% |

`scientificName` improves everywhere — non-US herbaria used printed labels far more
than 19th-century New England collectors. `recordedBy` falls 20 points or more because
its collector gazetteer is built from the GBIF eval set, so a collector outside it has
no reference.

Calibration held and was **conservative**: Anthropic's gap was -0.01, Google
under-claimed by 0.24. Nothing over-stated at the top of any range.

**Decision.** Unfamiliar specimens produce *less* confidence and more review, not
misplaced confidence. That is the failure mode we want. No change shipped.


## 4.5 Hand-labelled ground truth: a 40-specimen trial

**Question.** All ground truth comes from GBIF, and `scientificName`'s strongest
confidence signal is a GBIF backbone match, so that field is partly graded by its own
source. How much does that inflate the numbers?

**Method.** 40 specimens sampled across 40 institutions, labelled from the images by a
non-expert using `nbs/label_tool.py`, which shuffles pipeline candidates and hides
their source. Scored the pipelines against both truth sets.

**Result.** Against hand labels, Anthropic's `scientificName` appeared to fall from
97% to 72%. Examining all ten disagreements: **none were pipeline errors.** Four were
transcription slips by the labeller (`Gabzum triftorum` for *Galium triflorum*), two
were genus abbreviations (`S. subsecundum`), and four were synonyms where the pipeline
gave today's accepted name and the label gave what the sheet says.

The measured non-expert error rate on cursive species names is about **11%**.

**Decision.** No inflation detected — but not confirmed either. A truth set 11% wrong
cannot validate a pipeline claiming 85%; the noise exceeds the effect. Closing this
needs a botanist on the species column, roughly twenty minutes of expert time.

A deeper limit was recorded: `scientificName` **cannot be hand-labelled independently
even in principle**. Deciding whether *Jamesoniella autumnalis* (what the sheet says)
or *Syzygiella autumnalis* (today's accepted name) is correct requires a taxonomic
authority, and the authority is GBIF. The pipelines already store both, so reading and
normalisation should be scored as separate questions.


## 4.6 Correction to 1.7 and 2.4 — the fabrication metric was measuring capitalisation

**Question.** Re-measuring the invented-species rate on a larger common set gave a
different answer from 2.4. Which is right?

**Method.** Inspected every name counted as invented.

**Result.** They were real species with the epithet capitalised — `Rumex Acetosella`,
`Geranium Robertianum`, `Euphorbia Cyparissias L.` — a 19th-century labelling
convention. GBIF returns `NONE` for that form, so a correctly read name counted as
fabricated.

It penalised Claude hardest **because Claude transcribes the label faithfully** while
gpt-4o-mini silently normalises the case. The metric was rewarding the less accurate
behaviour.

Corrected rates on 239 specimens, after retrying each unmatched name with a lowercased
epithet:

| Pipeline | reported in 1.7 / 2.4 | corrected |
|---|---:|---:|
| Azure (gpt-4o-mini) | 11.3% | **12.3%** |
| Google | not measured | 6.0% |
| Claude Sonnet 5 | 1.3% | **0.8%** |

The gap between Anthropic and Azure is fifteenfold, not fourfold.

**Decision.** `GbifTaxonMatcher.match` retries with a lowercased epithet. This also
fixes the taxon correction and the `scientificName` confidence signal for those names,
both of which previously read `NONE` for a correct answer. **Sections 1.7 and 2.4
above are superseded by this table.**
