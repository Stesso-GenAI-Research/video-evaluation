# Data contracts

This folder describes optional experiment inputs. The main pipeline reads the
nested `indexed-videos-*.jsonl` export directly, so these older clip/step
contracts are not needed for normal local search.

Each JSONL file has one JSON object per line.

### `original_rankings.schema.json`

This is the important contract for the current comparison experiment. Each row
contains one query and the exact result order produced by the existing system:

```json
{"step_id":"step-1","query":"remove old faucet","original_matches":[{"clip_id":"clip-1","rank":1},{"clip_id":"clip-2","rank":2},{"clip_id":"clip-3","rank":3}]}
```

Ranks must start at 1 and be contiguous. Every row must have the same result
count as the command's `--top-k` value (default 3). A match can use a canonical
`clip_id` written by `build-index`, or the source `video_id`, `start_seconds`,
and `end_seconds`:

```json
{"video_id":"1451721","start_seconds":94.616,"end_seconds":118.3,"rank":1}
```

Timestamp references are resolved to canonical IDs with a 0.05-second default
tolerance. The command rejects missing and ambiguous matches before running
search; change the tolerance with `--timestamp-tolerance-seconds` when needed.

## Legacy generic pairwise contracts

The next three schemas belong to the older dense/pairwise pipeline. They are
kept for compatibility and are not needed by `build`, `search`, `compare`, or
the current automatic benchmark.

### `clips.schema.json`

This schema describes `clips.jsonl`. Each row represents one indexed video clip. The required field is `clip_id`. The row can also include project and video IDs, title, description, summary, transcript fields, `gemini_metadata`, timestamps, and dense embedding arrays.

### `steps.schema.json`

This schema describes `steps.jsonl`. Each row represents one project step. The required field is `step_id`. The row can also include project ID, step index, title, description, tools, materials, techniques, and dense embeddings.

### `pairwise.schema.json`

This schema describes `pairwise.jsonl`. Each row represents one human or model comparison between two clips for the same step. The required fields are `comparison_id`, `step_id`, `clip_a_id`, `clip_b_id`, and `winner_clip_id`.

Extra fields are allowed in all three schemas. This is useful because the export can preserve original Stesso metadata even if the current Python code does not use every field yet.
