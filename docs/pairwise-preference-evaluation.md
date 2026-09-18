# Pairwise preference evaluation (Project 1, Step 1)

For current result tables, exact development and W25 commands, output paths,
and project status, see the
[project status, experiments, and execution guide](stesso-project-update.md).
This page is the technical reference for the evaluator.

## Purpose

`evaluate-pairs` measures whether the frozen lexical, structured, or hybrid
scorer agrees with an already-observed A/B clip preference for a project step.
It scores only the supplied clips:

```text
score(step query, clip A)
score(step query, clip B)
```

It does not retrieve a challenger, build a review sheet, or infer a preference
from corpus rank.

For each method:

- higher score for the judged winner: credit `1.0`;
- exact score tie: credit `0.5`; and
- higher score for the judged loser: credit `0.0`.

Mean credit over resolved valid comparisons is the agreement rate.

## Inputs

The command accepts three inputs:

- `--index`: a nested `indexed-videos-*.jsonl` export or a built index directory;
- `--steps`: rows following
  [`preference_steps.schema.json`](../data_contracts/preference_steps.schema.json);
  and
- `--pairs`: rows following
  [`preference_pairs.schema.json`](../data_contracts/preference_pairs.schema.json).

The current preference contract is separate from the legacy
`pairwise.schema.json`, which belongs to the old-versus-new ranking workflow.

A clip reference may contain:

- a canonical `clip_id`; or
- `video_id`, `start_seconds`, and `end_seconds`.

Exact IDs resolve directly. Timestamp references use the shared canonical
segment resolver with a 0.05-second tolerance. When both forms are supplied,
they must identify the same clip. The output preserves the original reference,
canonical ID when resolved, and resolution status.

The provisional W25 loader accepts `id` as an alias for `comparison_id`, `id`
as an alias for a step's `step_id`, and `winner` as an alias for
`winner_position` only when its value is `A` or `B`. Output always uses canonical
field names. Each comparison requires nonblank `judge_provenance`; optional
`judge_confidence` must be numeric from 0 through 1.

## Frozen query and scoring configuration

The query contains nonempty step fields in this order:

1. `title`
2. `description`
3. `Tools: ` plus source-order tools joined by `, `
4. `Materials: ` plus source-order materials joined by `, `

Parts are joined with one space. Empty inventories add no labeled part. Every
scored row stores the final query and the run manifest stores the construction
rule.

The evaluator uses the existing production query parser and scorer defaults:

| Setting | Frozen value |
|---|---|
| Lexical | Existing TF-IDF configuration |
| Action weight | 0.55 |
| Object weight | 0.35 |
| Tool/supply weight | 0.10 |
| Hybrid alpha | 0.5 |
| Timestamp tolerance | 0.05 seconds |

A missing parsed action remains visible in diagnostics. It produces a structured
tie and the existing hybrid lexical fallback. Step 1 does not change parser
behavior, weights, alpha, thresholds, query formulation, or retrieval settings.

## Validation

The evaluator records, rather than silently discarding:

- malformed JSONL;
- missing or unknown step IDs;
- missing clip references;
- unresolved or ambiguous timestamps;
- invalid winner positions;
- the same canonical clip in both positions;
- a winner that cannot be mapped to a resolved clip; and
- malformed provenance or confidence.

Fatal file-level contract errors stop evaluation with a clear message. Row-level
resolution outcomes are written to `validation_report.json` and included in
total, resolved, and unresolved counts.

## Bootstrap confidence intervals

The evaluator uses a nonparametric clustered bootstrap. One iteration samples
`step_id` values with replacement and includes every comparison belonging to
each selected step. It does not sample comparison rows independently.

The default is 5,000 iterations and a user-supplied seed. The seed, iteration
count, cluster field, and confidence level are stored in the manifest and
summary. A fixed seed produces reproducible intervals.

## Generic command

```bash
./scripts/run_local_pipeline.sh evaluate-pairs \
  --index path/to/indexed-videos.jsonl \
  --steps path/to/steps.jsonl \
  --pairs path/to/pairwise.jsonl \
  --output project1_outputs/run-name/step1 \
  --seed 42 \
  --bootstrap-iterations 5000
```

Then create the descriptive case inventory:

```bash
./scripts/run_local_pipeline.sh analyze-pairs \
  --pair-scores project1_outputs/run-name/step1/pair_scores.jsonl \
  --output project1_outputs/run-name/diagnostics
```

`analyze-pairs` does not change scores. Its overlapping flags identify parse
failures, structured ties, zero structured evidence, method disagreements, and
method-specific agreement with the supplied winner.

## Evaluation outputs

The output directory contains:

| File | Contents |
|---|---|
| `pair_scores.jsonl` | One row per scorable comparison: query, original references, resolution, canonical IDs, judgment metadata, scores, predictions, credits, and structured diagnostics |
| `summary.json` | Resolution totals; method agreement, wins, ties, losses, intervals; parse, provenance, confidence, and winner-position counts |
| `summary.md` | Compact method table and frozen configuration |
| `validation_report.json` | File- and row-level validation and resolution outcomes |
| `manifest.json` | Inputs, hashes, query rule, scoring settings, bootstrap settings, and software provenance |

The summary reports total, resolved, unresolved, and resolution percentage. The
operational target for the authentic export is at least 90% resolved; the code
reports actual data quality and does not fabricate replacements.

## Diagnostic outputs

`analyze-pairs` writes:

| File | Contents |
|---|---|
| `diagnostic_cases.csv` | Auditable pair-level case table and overlapping flags |
| `diagnostic_summary.json` | Machine-readable descriptive counts |
| `diagnostic_summary.md` | Compact readable counts |
| `diagnostic_manifest.json` | Input hash and analysis provenance |

Diagnostics are inputs to later Step 2 work. They are not parser fixes, causal
error labels, or evidence that a method should be tuned.

## Interpretation

The result is preference agreement for supplied A/B pairs. It is not top-*k*
retrieval accuracy, parser precision, or evidence that the method would have
retrieved either clip from the full corpus.

All valid delivered judgments form the primary result. Provenance counts,
confidence values, winner positions, and optional provenance-specific summaries
remain descriptive. A smaller human subset is not used for tuning or promoted
to the primary result.

Generated development labels exercise the pipeline only. No primary Step 1
research conclusion is available until the authentic W25 index, steps, and
already-observed preference judgments are evaluated.
