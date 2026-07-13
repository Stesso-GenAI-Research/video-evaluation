# Data and experiment status

Updated July 13, 2026.

## Ready now

`indexed-videos-250.jsonl` is sufficient for building and exercising the search
system. The automated ingest audit found:

- 250 source videos and 1,703 raw clip annotations
- 3 invalid intervals with `end <= start`
- 1,700 valid source rows
- 36 duplicate timestamp groups containing 37 redundant rows
- 1,663 canonical playable segments
- 582 canonical segments with a description or goal for the held-out benchmark
- 1,241 canonical segments with parsed tool metadata

The parser retains all parent and clip fields, raw annotation variants,
alternate names, inventory alternatives/purposes, and source provenance. It
writes invalid rows separately instead of repairing them silently.

The repository can now produce:

- a canonical searchable corpus;
- a versioned/hash-recorded feature index;
- lexical, structured, and hybrid top-k search;
- a neutral one-query ranking diff;
- a validated batch comparison against supplied original rankings;
- a blinded relevance-review CSV;
- a scorer for completed blind reviews with Precision@k and paired intervals;
- a direct-title and exact-phrase controlled benchmark with confidence intervals.

## Current measured result

The held-out benchmark uses 582 candidates and 463 eligible queries. Candidate
titles and parent-video text are excluded.

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Lexical TF-IDF | 63.1% | 85.5% | 93.3% | 0.749 |
| Structured action | 26.1% | 35.4% | 47.5% | 0.334 |
| 50/50 hybrid | 50.5% | 61.6% | 68.0% | 0.577 |

Structured action matching is not yet competitive for whole-corpus retrieval.
Within a single source video, lexical and hybrid Hit@1 are both 72.6%, and the
paired confidence interval includes zero. This is a useful direction to study,
but not evidence that hybrid search is better.

## Data that is genuinely missing

The JSONL does not contain:

- project query/step IDs;
- the existing system’s original top-three results;
- retrieval ranks or scores;
- human relevance judgments;
- dense embeddings, captions, or transcripts.

Those fields are not needed to run search or the development benchmark. They
are needed to answer the stronger question, “Are the new top three actually
better than the old top three?”

The next supervisor export should contain one row per query with:

```json
{"step_id":"step-1","query":"remove old faucet","original_matches":[{"clip_id":"clip-1","rank":1},{"clip_id":"clip-2","rank":2},{"clip_id":"clip-3","rank":3}]}
```

If canonical IDs are unavailable, video IDs plus exact start/end timestamps are
enough to map results. A useful first human study would include 50–100 queries,
stratified across categories, with the old top three preserved in rank order.

## Next work

1. Obtain and run the batch of real original rankings.
2. Judge the pooled old/new clips using the generated blind worksheet.
3. Report Precision@3, action/object/tool agreement, wins/ties/losses, and paired
   confidence intervals.
4. Analyze the 106 narrative queries for which the current action parser still
   finds no action.
5. Tune parser/scorer changes on development videos only.
6. Freeze the configuration and evaluate on held-out videos.
7. Run the unchanged pipeline on the larger IndexedVideo export.

The current sample is no longer a blocker. The main missing evidence is an old
ranking plus human relevance judgments, not more extraction code.
