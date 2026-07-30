# OpenAI and Azure Cost Projection


This covers three things: what paid services we use and a 2 year
experimentation projection, a scoped plan for the next 3 months of experiments, and
a 2 year projection for the case where the experiments work and we start processing
real collections.


## What we use (As of July 2026)

| Service | Role | Status | Price (2026) |
|---|---|---|---|
| Azure Document Intelligence (prebuilt-read) | OCR of the label image | Production | $1.50 per 1,000 pages |
| OpenAI gpt-4o-mini, text | Extract 6 fields from the OCR text | Production | $0.15 / 1M in, $0.60 / 1M out |
| OpenAI gpt-4o-mini, self consistency and vision | The confidence signals | Production, on by default | same model |
| GBIF API | Taxonomy check and ground truth | Production | Free |
| Anthropic claude-3-7-sonnet | Alternate pipeline | Experimental | $3 / 1M in, $15 / 1M out |
| Google Document AI | Alternate OCR | Experimental, blocked on creds | about $1.50 per 1,000 pages |
| Tesseract | Free OCR baseline | Experimental | Free |

Only Azure plus gpt-4o-mini carry recurring cost. GBIF is free. The gpt-4o-mini
Batch API is half price ($0.075 in, $0.30 out) for bulk jobs that do not need an
instant answer, which matters at production scale (Haven't tested yet).

Prices as of July 2026 (see Sources).

## Unit cost

Cost per specimen for the default pipeline, anchored to what our cached test runs
actually cost, not to vendor estimates.

| Component | Calls | Cost |
|---|---:|---:|
| Azure prebuilt-read OCR | 1 | $0.0015 |
| gpt-4o-mini extraction | 1 | $0.0005 |
| Self consistency re-reads (K=3) | 3 | $0.0015 |
| Vision read | 1 | $0.0010 |
| Total | 6 | about $0.0045 |

Rounded for planning:

| Pipeline | per specimen | per 1,000 |
|---|---:|---:|
| Default (Azure) | $0.0045 | $4.50 |
| Default with Batch API | $0.0030 | $3.00 |


## (a) Two year experimentation projection

Assume a budget ceiling of $2,000 a month for testing, over 24 months, with no bulk
production runs.

In practice almost all of our iteration runs off cached results at no API cost, by
design. Building the entire 550 specimen test set cost about $2 in API. Re-scoring,
signal changes and calibration are free after that. We only pay when we build a new
labeled set or run a scale test on fresh images.

| | Monthly | 24 months |
|---|---:|---:|
| Budget ceiling | $2,000 | $48,000 |
| What we expect to spend | $50 to $200 | $1,200 to $4,800 |

A realistic active month is a fresh 1k labeled set (about $5), a couple of scale
tests on 10k to 20k images (about $50 to $100 each), and an occasional cross model
run (about $50). Call it $100 to $200 in a busy month and near zero in a quiet one.

Projection: budget $48k over two years, expect to actually spend $2k to $5k. The few
thousand a month figure is a comfortable ceiling and headroom for new ideas or testing.

## (b) The next three months of experiments (As of July 2026)

The open backlog, grouped by goal, each with its API cost. Most are free because
they use offline references, free APIs, or re-runs off existing caches.

Improve accuracy and confidence:

| Experiment | Why | Cost |
|---|---|---:|
| Barcode catalog lookup against GBIF occurrences | The one field with a real signal ceiling; 58% of wrong barcodes still score high | free |
| recordedBy external authority (Bionomia, Harvard Index) | Gives unknown collectors a reference so they stop scoring falsely low | free |
| Cross pipeline agreement as a signal | A second OCR engine gives a truly independent check, the kind needed to push rho past 0.5 | about $15 |
| Cross model bake off, including the newest frontier models (GPT-5.6, Opus 4.8, Sonnet 5, Fable 5) vs gpt-4o-mini | Pick the best accuracy per dollar, and see if any frontier model is worth its premium | about $100 to $250 |
| Vision detail sweep | See if a smaller or lower detail image keeps the lift and cuts cost | about $5 |

Prove it honestly:

| Experiment | Why | Cost |
|---|---|---:|
| Hand labeled test set, 200 to 500 non GBIF | The honest test that escapes GBIF circularity, and re-checks the existing calibration off circular data | about $9 |
| Out of domain robustness on 10k to 20k fresh images | Confirms accuracy holds outside New England | about $90 |

De-risk scale up:

| Experiment | Why | Cost |
|---|---|---:|
| Token logging | Replaces the modeled cost figures here with logged truth | free |
| Batch API validation run | Confirms the 50% saving and the latency tradeoff before any large run | about $20 |
| Prompt caching measurement | Confirms the cached input discount on the fixed prompt | about $5 |

Three month budget: likely $250 to $900 of API, with a ceiling of about $4,500 if we
run at the full $1,500 a month envelope. 


## (c) Two year projection if the experiments work

Assume the experiments pan out and we move from test set iteration to processing real
collections, still inside the few thousand a month idea from part (a).

Cost is driven by how many specimens we process, plus a small steady state for new
accessions. All figures use the default pipeline. The batch column is the
recommended cost for bulk jobs.

| Scenario | Specimens | Default | With Batch |
|---|---:|---:|---:|
| Test iteration, cached | 500 | about $0 | |
| New England corpus | 1.5M | $6,750 | $4,500 |
| Symbiota wide, aspirational | 10M | $45k | $30k |
| New accessions per year | 200k | $900 | $600 |

Two year build up for the New England target, using the Batch API:

| | Cost |
|---|---:|
| Year 1 pilot and first regional batches, about 500k processed | about $1,500 |
| Full New England corpus once, about 1.5M | about $4,500 |
| New accessions, two years at 200k a year | about $1,200 |
| Cross model and re-run experimentation, two years | about $2,000 |
| Two year total | about $9,000 to $15,000 |

Even the biggest single event, a full 1.5M run, is about $4,500 with the Batch API,
which is roughly two months of the envelope and happens once. Steady state after that
is well under $1,500 a month. If we ever go Symbiota wide at 10M or more, the one
time cost rises to about $30k with batching, which is the point to get a committed
tier Azure quote.

### If a frontier model wins the bake off

All the numbers above assume we ship gpt-4o-mini. If the bake off in part (b) shows a
frontier model is clearly more accurate and we decide to ship it, the premium hits
only the LLM calls, not Azure OCR, and we would likely drop the 3 self consistency
re-reads (a more accurate model does not need them). Full 1.5M New England corpus,
one time, with the Batch API:

| Extractor | per specimen | full 1.5M corpus | accessions per year (200k) |
|---|---:|---:|---:|
| gpt-4o-mini (current plan) | $0.003 | about $4,500 | about $600 |
| Sonnet 5 or GPT-5.6 Terra (about 20x) | ~$0.015 | about $20k | a few thousand |
| Opus 4.8 (about 30x) | ~$0.025 | about $30k to $40k | a few thousand |



## Sources

- Azure Document Intelligence, Read at $1.50 per 1,000 pages:
  https://learn.microsoft.com/en-us/answers/questions/1684935/pricing-azure-document-intelligence-service
- gpt-4o-mini at $0.15 / 1M in, $0.60 / 1M out, Batch half price:
  https://pricepertoken.com/pricing-page/model/openai-gpt-4o-mini
- Frontier model pricing (July 2026): OpenAI GPT-5.6 (Luna $1/$6, Terra $2.50/$15,
  Sol $5/$30) https://www.aipricing.guru/openai-pricing/ ; Anthropic Opus 4.8 $5/$25,
  Sonnet 5 $3/$15, Fable 5 $10/$50 (Anthropic pricing docs).
- Internal anchors: measured 550 specimen cache costs (Azure about $0.72, K=5 self
  consistency about $1.20); nbs/ocr_cache.py, nbs/self_consistency_cache.py,
  transcription/doc_intelligence.py.
