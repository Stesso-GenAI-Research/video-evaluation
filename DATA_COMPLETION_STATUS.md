# Data and experiment status

Updated July 13, 2026.

## Ready now

`indexed-videos-250.jsonl` is sufficient for building and exercising the search
system. The automated ingest audit found:

- 250 source videos and 1,703 raw clip annotations
- 3 invalid intervals with `end <= start`
- 1,700 valid clip annotations
- 36 duplicate timestamp groups containing 37 redundant rows
- 1,663 canonical playable segments
- 582 canonical segments with a description or goal for the field-held-out
  development benchmark
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

The field-held-out development benchmark uses 582 candidates and 462 eligible
queries. Candidate titles and parent-video text are excluded. It is a repeatable
development test, not a final held-out-video evaluation.

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Lexical TF-IDF | 63.2% | 85.5% | 93.3% | 0.749 |
| Structured action | 26.4% | 36.3% | 47.1% | 0.335 |
| 50/50 hybrid | 50.6% | 61.5% | 68.0% | 0.577 |

Structured action matching is not yet competitive for whole-corpus retrieval.
Within a single source video, lexical and hybrid Hit@1 are both 72.7%. The
paired confidence interval for the difference is approximately -4.9 to +5.4
points. This is a useful direction to study, but not evidence that hybrid
search is better.

These results require a positive score before a target can receive a rank. Exact
score ties receive expected rather than clip-ID-based credit. The structured
method has positive target evidence on 73.6% of queries and a positive target
tie on 45.7%, so coverage and coarse ties are major problems to improve.

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
4. Analyze the 107 narrative queries for which the current action parser still
   finds no action.
5. Tune parser/scorer changes on development videos only.
6. Freeze the configuration and evaluate on held-out videos.
7. Run the unchanged pipeline on the larger IndexedVideo export.

The current sample is no longer a blocker. The main missing evidence is an old
ranking plus human relevance judgments, not more extraction code.
