# OpenAI and Azure Cost Projection


This covers three things: what paid services we use and a 2 year
experimentation projection, a scoped plan for the next 3 months of experiments, and
a 2 year projection for the case where the experiments work and we start processing
real collections.


## What we use (As of August 2026)

All costs are dollars per 1,000 specimens, measured from real runs on our test set.

| Service | Role | Status | Cost per 1,000 |
|---|---|---|---|
| Azure Document Intelligence (prebuilt-read) | OCR of the label image | Production | $1.50 |
| OpenAI gpt-4o-mini, text | Extract 6 fields from the OCR text | Production | $0.50 |
| OpenAI gpt-4o-mini, self consistency and vision | The confidence signals, 4 extra calls | Production, on by default | $2.50 |
| GBIF API | Taxonomy check and ground truth | Production | Free |
| Anthropic Claude Sonnet 5 | Image-native alternate pipeline | Evaluated, recommended | $11.34 |
| Google Document AI | Alternate OCR | Experimental, blocked on creds | about $1.50 |
| Tesseract | Free OCR baseline | Experimental | Free |

Only Azure, OpenAI and Anthropic carry recurring cost. GBIF is free. The gpt-4o-mini
Batch API is half price for bulk jobs that do not need an instant answer, which
matters at production scale (haven't tested yet).

## Unit cost

What each pipeline costs per 1,000 specimens, from measured runs rather than vendor
estimates. Confidence scoring is priced separately because it is optional.

| Azure pipeline | Calls | Cost per 1,000 |
|---|---:|---:|
| Azure prebuilt-read OCR | 1 | $1.50 |
| gpt-4o-mini extraction | 1 | $0.50 |
| Self consistency re-reads (K=3) | 3 | $1.50 |
| Vision read | 1 | $1.00 |
| **Total** | **6** | **$4.50** |

| Anthropic pipeline | Calls | Cost per 1,000 |
|---|---:|---:|
| Claude Sonnet 5 extraction | 1 | $11.34 |
| Claude Sonnet 4.6 checker (2 confidence fields) | 1 | $6.56 |
| Azure OCR + gpt-4o-mini (1 confidence field) | 2 | $2.00 |
| **Total** | **4** | **$19.90** |

Rounded for planning:

| Pipeline | per 1,000 | notes |
|---|---:|---|
| Azure, extraction only | $2.00 | no confidence signals |
| Azure, full | $4.50 | confidence on 5 fields |
| Azure, full with Batch API | $3.00 | |
| Anthropic, extraction only | $11.34 | confidence on 2 fields |
| Anthropic, full | $19.90 | confidence on 5 fields |

### What the models cost

Measured on 150 specimens each, extraction only. Accuracy is the mean across the six
extracted fields.

| Model | Accuracy | Cost per 1,000 |
|---|---:|---:|
| Claude Opus 4.8 | 85% | $25.81 |
| **Claude Sonnet 5** | **85%** | **$11.34** |
| Claude Opus 4.6 | 79% | $10.78 |
| Claude Sonnet 4.6 | 75% | $6.56 |
| gpt-4o-mini (current production) | 74% | $4.50 |
| Claude Haiku 4.5 | 49% | $2.17 |

Sonnet 5 matches the most expensive model tested at less than half the price. Haiku
is not usable: it reads handwritten dates correctly only 39% of the time.


## (a) Two year experimentation projection

What it costs to keep experimenting, with no bulk production runs.

Almost all iteration runs off cached results at no API cost, by design. Building the
entire 550 specimen test set cost about $2. Re-scoring, signal changes and calibration
are free after that. We only pay when we build a new labeled set or run a scale test
on fresh images.

The best anchor is what work actually costs. Comparing six models across 530 labeled
specimens came to **$25** in total.

| | Cost |
|---|---:|
| Pre-purchase to get started | **$250** |
| Ongoing, per semester per team | **$25 to $50** |
| Ongoing, per year | **about $100** |

All three services are prepaid pay-as-you-go, so credits act as a hard ceiling: we
cannot overspend them, there is no subscription, and an idle month costs nothing.
Suggested split is $100 Anthropic, $50 OpenAI, $100 Azure. Azure and Microsoft both
run education grant programs that may cover Document Intelligence outright, which
would drop the total to about $150.

Costs stay low because the evaluation suite reuses cached results. Signal changes,
threshold tuning and calibration re-fits are free once the cache exists; we only pay
when processing new images.

The one thing that costs real money is processing an actual collection, and that is a
separate decision with a known price rather than an ongoing drain:

| | Cost |
|---|---:|
| 10,000-specimen pilot | $45 to $130 |
| Full New England corpus (1.5M), one time | $6,750 to $30,000 |

The range depends on which pipeline is used. Nothing above a pilot happens without
explicit approval.

## (b) The next three months of experiments (As of July 2026)

The open backlog, grouped by goal, each with its API cost. Most are free because
they use offline references, free APIs, or re-runs off existing caches.

Improve accuracy and confidence:

| Experiment | Why | Cost |
|---|---|---:|
| Barcode catalog lookup against GBIF occurrences | The one field with a real signal ceiling; 58% of wrong barcodes still score high | free |
| recordedBy external authority (Bionomia, Harvard Index) | Gives unknown collectors a reference so they stop scoring falsely low | free |
| ~~Cross pipeline agreement as a signal~~ | Done. Each pipeline now supplies the other's weakest confidence signals | spent $0 |
| ~~Cross model bake off~~ | Done. Sonnet 5 gives the best accuracy per dollar; no frontier model earned its premium | spent $25 |
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

Three month budget: likely **$100 to $150** of API, most of it the out of domain
robustness run. The $250 pre-purchase covers it.


## (c) Two year projection if the experiments work

Assume the experiments pan out and we move from test set iteration to processing real
collections, still inside the few thousand a month idea from part (a).

Cost is driven by how many specimens we process, plus a small steady state for new
accessions. The Azure column is the current pipeline at $4.50 per 1,000; the
Anthropic column is Sonnet 5 with full confidence at $19.90. Batch pricing halves
the Azure figure.

| Scenario | Specimens | Azure | Azure batched | Anthropic |
|---|---:|---:|---:|---:|
| Test iteration, cached | 500 | about $0 | | about $0 |
| 10,000-specimen pilot | 10k | $45 | $30 | $199 |
| New England corpus | 1.5M | $6,750 | $4,500 | $29,850 |
| Symbiota wide, aspirational | 10M | $45k | $30k | $199k |
| New accessions per year | 200k | $900 | $600 | $3,980 |

Two year build up for the New England target, using the Batch API:

| | Cost |
|---|---:|
| Year 1 pilot and first regional batches, about 500k processed | about $1,500 |
| Full New England corpus once, about 1.5M | about $4,500 |
| New accessions, two years at 200k a year | about $1,200 |
| Experimentation, two years | about $200 |
| Two year total | about $7,400 |

Even the biggest single event, a full 1.5M run, is about $4,500 with the Batch API,
and it happens once. Steady state after that is a few hundred dollars a year. If we
ever go Symbiota wide at 10M or more, the one time cost rises to about $30k with
batching, which is the point to get a committed tier Azure quote.

### What the pipeline choice costs

The bake off is done, so these are measured rather than estimated. Both pipelines
will be offered as a user choice, so the corpus figure depends on which one users
pick. Full 1.5M New England corpus, one time:

| Pipeline | per 1,000 | full 1.5M corpus | accessions per year (200k) |
|---|---:|---:|---:|
| Azure, batched | $3.00 | about $4,500 | about $600 |
| Azure, full | $4.50 | about $6,750 | about $900 |
| Claude Sonnet 5, extraction only | $11.34 | about $17,000 | about $2,300 |
| Claude Sonnet 5, full confidence | $19.90 | about $30,000 | about $4,000 |

The Anthropic pipeline costs about four times as much and is about 11 accuracy points
better, so the choice is a real tradeoff rather than an upgrade. A sensible default is
Azure for bulk work and Claude for collections where accuracy matters most.



## Sources

- Azure Document Intelligence, Read at $1.50 per 1,000 pages:
  https://learn.microsoft.com/en-us/answers/questions/1684935/pricing-azure-document-intelligence-service
- gpt-4o-mini list price, Batch half price:
  https://pricepertoken.com/pricing-page/model/openai-gpt-4o-mini
- Anthropic model list prices: Anthropic pricing docs. Sonnet 5 is at an introductory
  rate until 2026-08-31, after which its cost per 1,000 rises from $11.34 to about
  $17.51.
- Internal anchors: every cost per 1,000 in this document is measured from real token
  usage on our 530 specimen test set, not from vendor estimates. Reproduce with
  nbs/model_run.py (prints cost per 1,000 after each run) and nbs/model_compare.py.
- Accuracy figures: nbs/model_compare.py on 150 randomly sampled specimens; see
  docs/lab-notebook.md part 2.
