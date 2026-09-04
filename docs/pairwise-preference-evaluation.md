# Pairwise preference evaluation (Project 1, Step 1)

## Purpose

This command tests whether the repository's lexical, structured, or hybrid
scorer agrees with clip preferences that have already been observed. For every
valid pair associated with one project step, it directly computes each method's
score for clip A and clip B. It does not retrieve challengers or create a
blinded review sheet.

For each method, agreement credit is:

- `1.0` when the judged winner receives the higher score;
- `0.5` when A and B receive exactly equal scores; and
- `0.0` when the judged loser receives the higher score.

The primary result is the mean credit over all resolved, valid comparisons.

## Inputs

`evaluate-pairs` takes three JSONL files:

- `--index`: either the original nested `indexed-videos-*.jsonl` export or an
  index directory produced by `build` (the directory that contains `input/`,
  `month1/`, and `month2/`);
- `--steps`: project-step rows following
  [`preference_steps.schema.json`](../data_contracts/preference_steps.schema.json);
  and
- `--pairs`: judged comparisons following
  [`preference_pairs.schema.json`](../data_contracts/preference_pairs.schema.json).

Each `clip_a` and `clip_b` reference contains either a canonical `clip_id` or
`video_id`, `start_seconds`, and `end_seconds`. Exact IDs are resolved directly.
Timestamp references use the existing canonical segment resolver and a frozen
default tolerance of 0.05 seconds. If both forms are present, they must agree.
Missing and ambiguous references are recorded as unresolved; they are never
silently dropped or scored as if they had resolved.

The W25 field names were still provisional when this path was implemented. The
loader accepts `id` as an alias for a pair's `comparison_id`, and `winner` as an
alias for `winner_position` only when the value is `A` or `B`. A step may use
either `id` or `step_id`. Output uses the canonical names. Every comparison must
have nonblank `judge_provenance`; optional `judge_confidence` must be between 0
and 1.

## Frozen evaluation rule

This is a no-tuning evaluation. A step query is built once, deterministically,
from nonempty parts in this exact order:

1. `title`
2. `description`
3. `Tools: ` followed by tools in source order, joined with `, `
4. `Materials: ` followed by materials in source order, joined with `, `

The nonempty parts are joined with one space. Empty tool or material arrays do
not add their labelled part. The finalized query is written to every scored
row, and the construction rule is recorded in the run provenance. Structured
query parsing uses the same production query parser and imperative fallback as
search; the existing Month 1 inventory helper then attaches the supplied step
tools and materials as context to each parsed action. A missing action yields a
structured tie and the production hybrid method's lexical fallback, both of
which remain visible in row diagnostics.

The evaluation uses the repository's existing frozen scorer settings:

- unchanged TF-IDF lexical configuration;
- structured weights of 0.55 action, 0.35 object, and 0.10 tool/supply; and
- hybrid alpha 0.5 (equal lexical and structured contributions).

Do not select another query form, change parsing, sweep hybrid alpha, or adjust
weights, thresholds, retrieval settings, or timestamp tolerance after looking
at agreement results. Such work belongs to a later experiment, not Step 1.

## Run the evaluation

The direct W25 command, using the raw nested index export, is:

```bash
./scripts/run_local_pipeline.sh evaluate-pairs \
  --index data/indexed-videos-w25.jsonl \
  --steps data/steps-w25.jsonl \
  --pairs data/pairwise-w25.jsonl \
  --output project1_outputs/w25/step1 \
  --seed 42 \
  --bootstrap-iterations 5000
```

The iteration flag is optional; 5,000 is the default. The seed makes the
bootstrap confidence intervals reproducible.

If the W25 index has already been built, pass that index directory instead:

```bash
SAMPLE_JSONL=data/indexed-videos-w25.jsonl \
SAMPLE_OUTPUT_DIR=project1_outputs/w25/index \
  ./scripts/run_local_pipeline.sh build

./scripts/run_local_pipeline.sh evaluate-pairs \
  --index project1_outputs/w25/index \
  --steps data/steps-w25.jsonl \
  --pairs data/pairwise-w25.jsonl \
  --output project1_outputs/w25/step1 \
  --seed 42
```

Building first is optional; do not create another intermediate format solely
for this evaluator.

## Outputs

The output directory contains:

- `pair_scores.jsonl`: one auditable record per input comparison, preserving
  the original references, resolution status, canonical IDs, fixed query,
  judgment provenance and confidence, and A/B scores, predicted winner, and
  credit for all three methods. Existing structured diagnostics are included
  for later analysis, but Step 1 does not use them to modify scoring.
- `summary.json`: machine-readable resolution counts and percentage; each
  method's mean credit, wins, ties, losses, and 95% confidence interval; plus
  descriptive parse, provenance, confidence, and winner-position counts.
- `summary.md`: a concise human-readable method table and frozen run
  configuration.
- `validation_report.json`: row-level input and clip-resolution outcomes,
  including malformed, unknown-step, unresolved, ambiguous, same-clip, and
  inconsistent-winner errors. Comparisons that cannot be scored remain
  visible here and in the totals.
- `manifest.json`: input and configuration provenance needed to reproduce the
  run.

The summary always distinguishes total, resolved, and unresolved comparisons
and reports the resolution percentage. The expected operational target for the
real export is at least 90% resolution, but the evaluator reports what the data
actually contains rather than manufacturing replacements.

## Confidence intervals and provenance

The 95% intervals use a nonparametric clustered bootstrap. A draw samples
`step_id` clusters with replacement and includes all comparisons belonging to
each selected step. It never samples pair rows independently, because judgments
for the same step need not be independent. Both the seed and number of draws
are stored with the outputs.

All delivered valid judgments form the primary result. Provenance counts and
optional descriptive cascade/human slices are retained for interpretation;
the smaller human subset is not used for tuning or promoted to the primary
result.

## Interpretation

The reported number is preference agreement: given the same judged A/B pair,
how often did a frozen scoring method favor the observed winner? It is not
top-*k* accuracy, corpus retrieval recall, parser quality, or proof that one
method will retrieve better candidates from the full index.

Synthetic fixtures exercise the machinery only. No research conclusion should
be recorded until the real Stesso W25 index, steps, and preference export have
been evaluated. Error analysis, parser changes, tie reduction, alpha sweeps,
reranking experiments, and other optimization are explicitly deferred to
Steps 2 and 3.
