# Project status, experiments, and execution guide

Last verified: September 18, 2026.

This document is the source of truth for current project status, data inventory,
experiment results, reproduction commands, output locations, and the Stesso
project update. Method details and data contracts remain in the linked technical
references.

## 1. Current status

The repository can:

- build a canonical timestamped clip index from the nested IndexedVideo export;
- extract action, object, tool, and supply evidence;
- score clips with frozen lexical, structured, and 50/50 hybrid methods;
- run the field-held-out retrieval development benchmark;
- resolve clip IDs or video/timestamp references to canonical clips;
- directly score already-judged A/B pairs;
- validate every pair without silently dropping invalid rows;
- compute deterministic 95% bootstrap intervals clustered by `step_id`; and
- produce pair-level outputs and descriptive disagreement inventories.

The primary W25 pairwise-preference experiment is implemented.

## 2. Research question and frozen methods

The primary question is:

> When a judge preferred one clip over another for the same project step, does
> action-object-tool matching agree with that choice more often than lexical
> matching?

Each valid pair is scored directly. A method receives credit `1.0` when it
scores the observed winner higher, `0.5` for an exact tie, and `0.0` when it
scores the loser higher. Credits are averaged by method. Confidence intervals
resample complete `step_id` clusters rather than individual pair rows.

The Step 1 configuration is frozen:

| Component | Fixed setting |
|---|---|
| Lexical | Existing TF-IDF configuration |
| Structured weights | action 0.55, object 0.35, tool/supply 0.10 |
| Hybrid | alpha 0.5 |
| Query fields | title, description, tools, materials, in that order |
| Timestamp tolerance | 0.05 seconds |
| Bootstrap | 5,000 iterations by default, clustered by `step_id` |

Step 1 does not tune parsing, weights, alpha, thresholds, query formulation, or
retrieval settings.

## 3. Data inventory

### 3.1 Real indexed-video sample

The current real sample contains:

| Item | Count |
|---|---:|
| Parent videos | 250 |
| Raw nested clip annotations | 1,703 |
| Valid annotations | 1,700 |
| Rejected invalid intervals | 3 |
| Canonical searchable clips | 1,663 |
| Action records | 8,899 |
| Clips with at least one action | 1,296 / 1,663 (77.9%) |
| Action records with an object | 5,699 / 8,899 (64.0%) |
| Action records with a directly attached tool | 661 / 8,899 (7.4%) |

These are coverage measurements, not parser-precision estimates. The 60-row
extraction review remains unlabeled.

### 3.2 Constructed pairwise development corpora

Two development corpora were generated from the real canonical index:

| Corpus | Steps | Pairs | Videos | Categories | Pair selection |
|---|---:|---:|---:|---:|---|
| Metadata contrast | 800 | 2,395 | 209 | 26 | Parsed metadata only |
| Paraphrased hard negative | 350 | 1,309 | 149 | Frozen scorer hard negatives |
| Total inventory | 1,150 | 3,704 | — | 26 | Do not aggregate rates |

Metadata-contrast pair types:

| Type | Pairs |
|---|---:|
| Same action, different object | 705 |
| Same object, different action | 687 |
| Within-video adjacent | 784 |
| Tool/material context contrast | 219 |

Paraphrased hard-negative pair types:

| Type | Pairs |
|---|---:|
| Lexical hard negative | 350 |
| Structured hard negative | 350 |
| Within-video adjacent | 350 |
| Same object, different action | 259 |

All 3,704 rows resolved to canonical clips. Validation recorded no fatal errors,
resolution errors, or dropped comparisons.

Both corpora use constructed target labels. Their manifests record
`human_judgments: false` and `research_evidence: false`. They support regression,
stress testing, and case selection only. They must not be described as authentic
human or production-cascade judgments.

## 4. Current experiment results

### 4.1 Field-held-out retrieval development benchmark

The benchmark evaluates 462 eligible queries against 582 candidates. Candidate
text excludes the clip name and parent-video text. Exact normalized phrase
leakage and ambiguous query names are excluded.

Whole-corpus retrieval:

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Lexical TF-IDF | 63.2% | 85.5% | 93.3% | 0.749 |
| Structured | 26.4% | 36.3% | 47.1% | 0.335 |
| Hybrid 50/50 | 50.6% | 61.5% | 68.0% | 0.577 |

Hybrid minus lexical Hit@1 is -12.6 percentage points. The video-clustered 95%
interval is approximately [-18.4, -6.9] points.

Within-video retrieval:

| Method | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| Lexical TF-IDF | 72.7% | 92.9% | 0.834 |
| Structured | 50.4% | 68.5% | 0.597 |
| Hybrid 50/50 | 72.7% | 92.5% | 0.833 |

The 95% interval for hybrid minus lexical Hit@1 is approximately [-4.9, +5.4]
points. This result does not establish equivalence.

### 4.2 Metadata-contrast pairwise benchmark

| Method | Resolved pairs | Agreement | Step-clustered 95% CI | Wins | Ties | Losses |
|---|---:|---:|---:|---:|---:|---:|
| Lexical | 2,395 | 99.83% | 99.66–99.96% | 2,391 | 0 | 4 |
| Structured | 2,395 | 95.91% | 95.08–96.72% | 2,225 | 144 | 26 |
| Hybrid | 2,395 | 99.75% | 99.54–99.92% | 2,389 | 0 | 6 |

This corpus is score-independent, but target metadata supplies both the query
and assigned winner. The result measures recovery of a construction rule, not
observed preference agreement.

### 4.3 Paraphrased hard-negative stress benchmark

| Method | Resolved pairs | Agreement | Step-clustered 95% CI | Wins | Ties | Losses |
|---|---:|---:|---:|---:|---:|---:|
| Lexical | 1,309 | 96.72% | 95.36–97.92% | 1,266 | 0 | 43 |
| Structured | 1,309 | 50.76% | 48.24–53.31% | 458 | 413 | 438 |
| Hybrid | 1,309 | 72.80% | 70.94–74.73% | 953 | 0 | 356 |

This corpus deliberately uses frozen scorer outputs to select hard negatives.
It is selection-biased and should be treated as a stress test.

### 4.4 Diagnostic inventory

Across the two constructed corpora:

| Diagnostic | Rows |
|---|---:|
| Lexical/structured prediction disagreement | 1,030 |
| Structured ties | 557 |
| Zero structured evidence | 415 |
| Structured-only correct under constructed label | 18 |
| Lexical-only correct under constructed label | 455 |
| Hybrid recovery | 15 |
| Hybrid regression | 330 |

Rates are not pooled because the corpora use different selection mechanisms.
The present evidence does not show structured scoring outperforming lexical
TF-IDF. Missing structured evidence and ties remain material. No Step 2 parser
or tie-reduction changes have been made from these observations.

## 5. Reproduce the current experiments

Run all commands from the repository root.

### 5.1 Setup and verification

```bash
./scripts/run_local_pipeline.sh setup
./scripts/run_local_pipeline.sh test
```

Build the current real sample index when needed:

```bash
./scripts/run_local_pipeline.sh build
```

### 5.2 Metadata-contrast development run

```bash
./scripts/run_local_pipeline.sh generate-contrast-pairs \
  --index project1_outputs/indexed-video-sample \
  --output project1_outputs/metadata-contrast-preference-development \
  --step-count 800 \
  --seed 42

./scripts/run_local_pipeline.sh evaluate-pairs \
  --index project1_outputs/indexed-video-sample \
  --steps project1_outputs/metadata-contrast-preference-development/steps.jsonl \
  --pairs project1_outputs/metadata-contrast-preference-development/pairwise.jsonl \
  --output project1_outputs/metadata-contrast-preference-development/step1 \
  --seed 42 \
  --bootstrap-iterations 5000

./scripts/run_local_pipeline.sh analyze-pairs \
  --pair-scores project1_outputs/metadata-contrast-preference-development/step1/pair_scores.jsonl \
  --output project1_outputs/metadata-contrast-preference-development/diagnostics
```

### 5.3 Paraphrased hard-negative development run

```bash
./scripts/run_local_pipeline.sh generate-controlled-pairs \
  --index project1_outputs/indexed-video-sample \
  --output project1_outputs/paraphrased-preference-development \
  --step-count 350 \
  --seed 42

./scripts/run_local_pipeline.sh evaluate-pairs \
  --index project1_outputs/indexed-video-sample \
  --steps project1_outputs/paraphrased-preference-development/steps.jsonl \
  --pairs project1_outputs/paraphrased-preference-development/pairwise.jsonl \
  --output project1_outputs/paraphrased-preference-development/step1 \
  --seed 42 \
  --bootstrap-iterations 5000

./scripts/run_local_pipeline.sh analyze-pairs \
  --pair-scores project1_outputs/paraphrased-preference-development/step1/pair_scores.jsonl \
  --output project1_outputs/paraphrased-preference-development/diagnostics
```

### 5.4 Frozen W25 run

When the three authentic files are available:

```bash
./scripts/run_local_pipeline.sh evaluate-pairs \
  --index data/indexed-videos-w25.jsonl \
  --steps data/steps-w25.jsonl \
  --pairs data/pairwise-w25.jsonl \
  --output project1_outputs/w25/step1 \
  --seed 42 \
  --bootstrap-iterations 5000

./scripts/run_local_pipeline.sh analyze-pairs \
  --pair-scores project1_outputs/w25/step1/pair_scores.jsonl \
  --output project1_outputs/w25/diagnostics
```

No tuning or preliminary W25 sweep should precede this command.

## 6. Implementation and output locations

### 6.1 Repository map

| Path | Responsibility |
|---|---|
| [`scripts/run_local_pipeline.sh`](../scripts/run_local_pipeline.sh) | Supported shell entry point |
| [`src/action_semantics/cli.py`](../src/action_semantics/cli.py) | CLI command definitions |
| [`src/action_semantics/retrieval/preference_evaluation.py`](../src/action_semantics/retrieval/preference_evaluation.py) | Input loading, frozen direct scoring, validation, bootstrap, and outputs |
| [`src/action_semantics/retrieval/clip_resolution.py`](../src/action_semantics/retrieval/clip_resolution.py) | Shared canonical-ID and timestamp resolution |
| [`src/action_semantics/retrieval/scorers.py`](../src/action_semantics/retrieval/scorers.py) | Structured and hybrid scoring |
| [`src/action_semantics/retrieval/lexical.py`](../src/action_semantics/retrieval/lexical.py) | Lexical TF-IDF scoring |
| [`src/action_semantics/contrast_preferences.py`](../src/action_semantics/contrast_preferences.py) | Score-independent metadata-contrast generator |
| [`src/action_semantics/synthetic_preferences.py`](../src/action_semantics/synthetic_preferences.py) | Synthetic and paraphrased hard-negative generators |
| [`src/action_semantics/pair_diagnostics.py`](../src/action_semantics/pair_diagnostics.py) | Descriptive pair diagnostics |
| [`data_contracts/`](../data_contracts/) | Current and legacy JSON schemas |
| [`tests/`](../tests/) | Unit and CLI integration tests |

### 6.2 Generated output map

A completed development corpus has this layout:

```text
project1_outputs/metadata-contrast-preference-development/
├── steps.jsonl
├── pairwise.jsonl
├── pair_audit.csv
├── DATA_NOTICE.md
├── generation_manifest.json
├── step1/
│   ├── pair_scores.jsonl
│   ├── summary.json
│   ├── summary.md
│   ├── validation_report.json
│   └── manifest.json
└── diagnostics/
    ├── diagnostic_cases.csv
    ├── diagnostic_summary.json
    ├── diagnostic_summary.md
    └── diagnostic_manifest.json
```

The paraphrased corpus uses the same layout under
`project1_outputs/paraphrased-preference-development/`. W25 outputs will be
written under `project1_outputs/w25/`.

Key files:

| File | Purpose |
|---|---|
| `pair_scores.jsonl` | Original references, canonical IDs, resolution status, query, judgment metadata, all scores, predicted winners, credits, and structured diagnostics |
| `summary.json` | Machine-readable resolution, agreement, confidence intervals, provenance, confidence, and winner-position counts |
| `summary.md` | Compact method table and frozen configuration |
| `validation_report.json` | Every malformed, unknown, unresolved, ambiguous, same-clip, or inconsistent-winner outcome |
| `manifest.json` | Input and configuration provenance |
| `diagnostic_cases.csv` | Pair-level descriptive flags and score differences for case review |
| `diagnostic_summary.json` | Aggregate diagnostic counts |

`project1_outputs/` is generated locally and ignored by Git. Regenerate it from
the versioned code and the corresponding private inputs.

## 7. Required W25 inputs

The frozen primary experiment requires:

1. `indexed-videos-w25.jsonl`, or an existing canonical index built from it;
2. `steps-w25.jsonl` with stable step ID, title, description, tools, and
   materials; and
3. `pairwise-w25.jsonl` with comparison ID, step ID, both clip references,
   winner position, judge provenance, and optional confidence.

Clip references may use a canonical `clip_id` or `video_id`, `start_seconds`,
and `end_seconds`. Before the full run, Stesso should provide:

- a small schema sample;
- definitions for judge-provenance values;
- confidence scale semantics; and
- confirmation whether repeated judgments are independent rows or aggregated.

The resolver reports missing and ambiguous references. The operational target
is at least 90% resolution, but invalid rows remain visible and are never
replaced or silently discarded.

## 8. Stesso meeting update

Use the following concise status:

> The indexing, frozen pairwise evaluator, timestamp resolution, validation,
> clustered bootstrap, and audit outputs are complete. We have 3,704
> constructed development comparisons across 1,150 steps, all of which resolve
> cleanly, plus a field-held-out retrieval benchmark on the real 250-video
> sample. Those development results currently favor lexical matching and expose
> structured-evidence coverage and tie problems. They are not human preference
> results. The remaining dependency for the primary research question is the
> authentic W25 steps and already-observed A/B judgments with provenance and
> confidence. Once delivered, the frozen experiment runs with one command and
> does not require method tuning.

Report the current negative result directly: structured scoring has not beaten
lexical scoring in either the real retrieval development benchmark or the
constructed pairwise benchmarks. The constructed datasets establish operational
readiness and provide diagnostic coverage; they do not substitute for authentic
preference data.

## 9. Next work

1. Validate a small W25 schema sample without changing frozen settings.
2. Run the frozen Step 1 evaluator once on all valid delivered judgments.
3. Report primary all-judgment agreement with provenance slices as descriptive
   secondary results.
4. Only after Step 1 is recorded, begin Step 2 error analysis on the existing
   diagnostic cases.
5. Reserve parser changes, tie reduction, alpha sweeps, and reranking work for
   declared development data in later steps.

## 10. Verification

The latest complete check produced:

```text
93 tests passed
All Ruff checks passed
Python source and tests compiled
Bash syntax check passed
CLI help passed for generate-contrast-pairs, evaluate-pairs, and analyze-pairs
```

See [Current verification record](verification.md) for test scope. See
[Pairwise preference evaluation](pairwise-preference-evaluation.md) for contract,
resolution, validation, and bootstrap details. See
[Running the pipeline](running-the-pipeline.md) for the full command reference.
