# Data contracts

A data contract describes the exact shape that an input file must have. These
files use JSONL: each line is one complete JSON object. Commands validate input
before using it; the preference evaluator records row-level failures in its
validation report, while older experiment commands fail fast.

The main search pipeline reads the nested `indexed-videos-*.jsonl` export
directly. The most important extra contract is the original-ranking input used
for the real comparison experiment.

Project 1 Step 1 has separate current contracts for evaluating preferences that
have already been judged. They intentionally do not replace the older generic
pairwise files described at the end of this page.

## `preference_pairs.schema.json`

This is the current contract for one already-judged A/B clip comparison. The
canonical fields are:

- `comparison_id`: a nonempty identifier unique within the input;
- `step_id`: the project step shared by both clips;
- `clip_a` and `clip_b`: nested canonical-ID or timestamp references;
- `winner_position`: exactly `A` or `B`;
- `judge_provenance`: a nonempty description of the judgment source;
- optional `judge_confidence`: a number from 0 through 1; and
- optional `winner_clip_id`: an additional consistency check when the source
  export records it.

A pair using timestamp references looks like this:

```json
{"comparison_id":"comparison-1","step_id":"step-1","clip_a":{"video_id":"1451721","start_seconds":94.616,"end_seconds":118.3},"clip_b":{"video_id":"1451721","start_seconds":61.383,"end_seconds":79.466},"winner_position":"A","judge_provenance":"cascade","judge_confidence":0.91}
```

An exact canonical reference instead uses `{"clip_id":"..."}`. A reference
may contain both forms; when it does, they must resolve to the same clip. The
evaluator maps timestamps to the canonical index with a 0.05-second default
tolerance and explicitly reports missing or ambiguous matches. It also rejects
same-clip pairs and a supplied `winner_clip_id` that does not identify the
resolved winner.

The W25 source field names were not final when this contract was added. As a
limited ingestion convenience, the loader also accepts `id` as an alias for
`comparison_id`, and `winner` as an alias for `winner_position` only when its
value is `A` or `B`. Output always uses the canonical names. Provenance is not
inferred: `judge_provenance` remains required.

## `preference_steps.schema.json`

This is the current step-text contract for preference evaluation. Each row has
either `id` (the expected W25 spelling) or `step_id`, plus `title`,
`description`, `tools`, and `materials`. Tools and materials are arrays of
strings. If both identifier spellings are supplied, they must agree.

The evaluator normalizes either identifier spelling to `step_id`. Extra source
metadata is retained at the contract boundary but is not used to adapt the
fixed Step-1 query.

## `original_rankings.schema.json`

This file describes one query and the existing system's exact ranked results.
Here is a complete three-result row:

```json
{"step_id":"step-1","query":"remove old faucet","original_matches":[{"clip_id":"indexed-video-1451721-segment-94p616-118p3","rank":1},{"clip_id":"indexed-video-1451721-segment-61p383-79p466","rank":2},{"clip_id":"indexed-video-1451721-segment-79p466-94p616","rank":3}]}
```

This example shows the format; it is not a claim that these were the private
system's historical results. The real export must preserve the real order.

The batch command checks that:

- every `step_id` is nonempty and unique in the file;
- every `query` is nonempty;
- every row contains exactly the command's `--top-k` result count;
- ranks start at 1 and are contiguous;
- one query does not repeat the same clip reference; and
- every reference maps to exactly one canonical clip before search begins.

The default `--top-k` is 3. If an experiment genuinely uses another value, pass
the same value to `compare-batch`.

### Referencing clips by timestamp

If the existing system does not store the new canonical clip IDs, each match
may instead use the source video ID and exact interval:

```json
{"video_id":"1451721","start_seconds":94.616,"end_seconds":118.3,"rank":1}
```

The command maps this reference to the canonical index with a 0.05-second
default tolerance. It rejects references that are missing or match more than
one segment. Change the tolerance only when the source systems round timestamps
differently:

```bash
./scripts/run_local_pipeline.sh compare-batch original_rankings.jsonl \
  --timestamp-tolerance-seconds 0.1
```

See [Running the pipeline](../docs/running-the-pipeline.md#comparing-rankings)
for the complete blinded-review workflow.

## Legacy generic pairwise contracts

The remaining schemas belong to an older dense/pairwise experiment path. They
are kept for compatibility. They are not needed by `build`, `search`,
`compare`, `compare-batch`, or the current automatic benchmark.

### `clips.schema.json`

This schema describes a generic `clips.jsonl`. Each row represents one indexed
video clip and requires `clip_id`. Optional fields include project and video
IDs, title, description, summary, transcripts, Gemini metadata, timestamps, and
dense embedding arrays.

### `steps.schema.json`

This schema describes a generic `steps.jsonl`. Each row represents one project
step and requires `step_id`. Optional fields include the project ID, step index,
title, description, tools, materials, techniques, and dense embeddings.

### `pairwise.schema.json`

This schema describes human or model comparisons between two clips for the same
step. It requires `comparison_id`, `step_id`, `clip_a_id`, `clip_b_id`, and
`winner_clip_id`.

These three legacy schemas allow extra fields so a larger export can retain
source metadata that the older experiment code does not use directly.
They remain unchanged for compatibility. In particular,
`pairwise.schema.json` uses flat `clip_a_id`/`clip_b_id` fields and is not the
input contract for the new `evaluate-pairs` command.
