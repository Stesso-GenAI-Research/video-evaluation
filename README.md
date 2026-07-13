# Action Semantics for Instructional Video Search

## The point of this project

I am trying to answer one practical question:

> If someone asks for a step such as “remove the old faucet,” can I return clips
> that perform that exact action on that exact object?

Normal text search is good at finding the same general topic. It is not always
good at distinguishing `remove faucet` from `install faucet`, or `tighten screw`
from another step that merely mentions screws. This project tests whether an
explicit action–object–tool/supply representation can improve that search.

There are now three working search methods:

- `lexical`: ordinary TF-IDF word matching
- `structured`: matching aligned action–object–tool/supply records
- `hybrid`: a 50/50 combination of lexical and structured scores

The research question is not settled in advance. The code now runs the
experiment and reports when the structured method loses as well as when it
wins.

## Where the project is now

The project has functional search, functional result-set comparison, a batch
comparison path for real old results, and a repeatable automatic benchmark.

I also corrected an important data mistake. The file has 1,703 raw clip
annotations, but those are not 1,703 distinct playable segments:

```text
1,703 raw clip rows
    3 rejected because end <= start
1,700 valid rows
   37 repeated rows merged across 36 timestamp groups
1,663 canonical searchable segments
```

Canonical IDs now come from the video ID and timestamps. They do not change if
the source list is reordered. Duplicate metadata is retained as provenance and
alternate names are kept as aliases. The three invalid intervals are written
to `input/rejected_clips.jsonl` instead of being silently repaired.

The parser also preserves every parent video field, including the source,
category ID and name, summary, goal, views, likes, comments, subscribers, URL,
and YouTube ID. Long tool/supply annotations are parsed into item names,
alternatives, and purposes while their raw text is kept.

## Run everything

Python 3.11–3.13 is required. Run setup once:

```bash
./scripts/run_local_pipeline.sh setup
```

The normal reproducible run is:

```bash
./scripts/run_local_pipeline.sh all
```

`all` runs the tests, rebuilds the canonical index from
`indexed-videos-250.jsonl`, verifies the generated artifacts, and runs the
benchmark. It only runs networked setup if the virtual environment is missing.

The main outputs are under:

```text
project1_outputs/indexed-video-sample/
```

Important files include:

- `input/indexed_video_profile.json`: source coverage and data-quality counts
- `input/indexed_video_clips.jsonl`: the 1,663 canonical search records
- `input/rejected_clips.jsonl`: the three invalid source annotations
- `index_manifest.json`: input hash, versions, field policy, and output hashes
- `benchmark/benchmark_summary.json`: automatic experiment results
- `benchmark/benchmark_queries.csv`: one row per evaluated query
- `quality/manual_review_sample.csv`: 60 extraction examples for manual checking

## Run a search

The easiest command uses the hybrid method and returns three clips:

```bash
./scripts/run_local_pipeline.sh search "remove old faucet"
```

Any search option after the query is passed to the Python program:

```bash
./scripts/run_local_pipeline.sh search "paint wall" \
  --method lexical \
  --top-k 5 \
  --max-per-video 1
```

Useful method choices are:

```bash
--method lexical
--method structured
--method hybrid
```

Each result contains the canonical clip ID, video ID, title, URL, start/end
timestamps, total score, and separate lexical, action, object, tool, supply,
and location/scope diagnostic signals. Zero-score clips are not returned just
to fill the requested number of results. If the action parser misses a
noun-like query, hybrid search falls back to lexical search and records a
warning instead of crashing.

Examples from the current index are encouraging at a qualitative level:

- `remove old faucet` returns `Remove Faucet Stem` and `Remove Faucet Handle`
  first.
- `paint wall` returns `Prepare and Paint Closet Walls` first.
- `paint wall with primer` returns the same clip with a supply-context match of
  1.0, showing that the parsed supply inventory is used by search.
- `tighten screw with screwdriver` returns `Tighten the Set Screws` first.

These examples prove that search runs. They do not prove overall accuracy.

## How the structured score works

For a sentence such as:

```text
Tighten the set screw with a screwdriver.
```

the parser creates roughly:

```text
action = tighten
object = screw
tool   = screwdriver
```

The scorer compares one complete query triple with one complete candidate
triple. It cannot take `tighten` from one sentence and `screw` from an unrelated
sentence to manufacture a strong match. Object and tool evidence only helps
when the actions are compatible, and positive and negated actions do not match.

The current structured score gives 55% of its weight to the action, 35% to the
object, and 10% to tool-or-supply context. If the query does not name an object
or context item, the unused weight is redistributed over the parts that are
present.

There is one deliberate compromise. A phrase such as `with a screwdriver`
names a tool, while `with primer` names a consumable supply. The dependency
parser cannot always tell those apart, so the scorer checks the phrase against
both the clip's tool inventory and its supply inventory and uses the better
match. Phrases such as `on the wall` or `into the housing` are reported as a
location/scope diagnostic, but they do not affect the production score. This
keeps locations from being incorrectly compared with material inventories.

VerbNet and FrameNet are used as weaker fallbacks when verbs differ. The early
DIY taxonomy is still generated for research, but it is not used for ranking;
its clusters have not been manually validated and sometimes group unrelated or
opposite actions.

## What the automatic experiment found

The current JSONL does not contain human relevance labels, so I built a careful
development benchmark from fields that are already aligned:

```text
query        = clip name
known target = the same timestamped clip
candidate    = description + goal + tools + supplies
```

The clip name and all parent-video text are removed from candidate documents.
This prevents the benchmark from giving the answer directly to the search
method. Duplicate query names, exact phrase leaks, and unparseable action
queries are excluded, but those clips remain in the candidate pool as
distractors.

The final benchmark has 582 candidates and 462 eligible queries. It uses the
same candidates and queries for all methods.

A target must receive a positive score to receive a rank. If several clips have
the exact same score, the metric averages over their possible tied positions;
it does not let alphabetical clip IDs decide whether the answer is correct.
This matters because the structured method produced a positive target score
for only 73.6% of queries and tied the target with another clip on 45.7% of all
queries. That is useful failure information, not something to hide with an
arbitrary tie-break.

| Whole-corpus method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Lexical TF-IDF | 63.2% | 85.5% | 93.3% | 0.749 |
| Structured action | 26.4% | 36.3% | 47.1% | 0.335 |
| 50/50 hybrid | 50.6% | 61.5% | 68.0% | 0.577 |

The honest finding is that structured-only retrieval is currently much weaker
than ordinary text search. The 50/50 hybrid is also worse globally. Its Hit@1
is 12.6 percentage points below TF-IDF, and the video-cluster bootstrap 95%
confidence interval is approximately -18.4 to -6.9 points.

I also ran a topically controlled within-video reranking task. It has a smaller
candidate set, but the alternatives come from the same source video:

| Within-video method | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| Lexical TF-IDF | 72.7% | 92.9% | 0.834 |
| Structured action | 50.4% | 68.5% | 0.597 |
| 50/50 hybrid | 72.7% | 92.5% | 0.833 |

The observed hybrid and lexical Hit@1 values are equal. The 95% interval for
the difference is about -4.9 to +5.4 points. That interval allows both modest
harm and modest benefit, so this experiment detected no reliable difference;
it did not prove that the methods are equivalent. Within-topic reranking is
still the more promising place to continue testing action features.

This benchmark uses a paired field as weak ground truth. It is useful for
development, but another clip could also be a valid answer. Only human review
can decide overall relevance in that situation. The clip names and descriptions
were also generated together, so they can still share partial wording or
paraphrases after exact phrases are removed. This controls the most direct
leakage; it does not make the fields fully independent.

## Compare two result sets

For one query, this command compares the lexical baseline with the new hybrid
ranking:

```bash
./scripts/run_local_pipeline.sh compare "remove old faucet"
```

The report shows overlap, set differences, and rank changes. It deliberately
sets `quality_claim` to `false` and `winner` to `null`. A method cannot prove it
is better by grading results with the same score that selected them.

If I have the exact original clip IDs, I can preserve and compare that real
order:

```bash
./scripts/run_local_pipeline.sh compare "remove old faucet" \
  --original-clip-id indexed-video-1451721-segment-94p616-118p3 \
  --original-clip-id indexed-video-1451721-segment-61p383-79p466
```

For a real experiment over many steps, create JSONL rows like this:

```json
{"step_id":"step-1","query":"remove old faucet","original_matches":[{"clip_id":"indexed-video-1451721-segment-94p616-118p3","rank":1},{"clip_id":"indexed-video-1451721-segment-61p383-79p466","rank":2},{"clip_id":"indexed-video-1451721-segment-79p466-94p616","rank":3}]}
```

Then run:

```bash
./scripts/run_local_pipeline.sh compare-batch path/to/original_rankings.jsonl
```

Each row must contain the same number of original results as `--top-k` (three by
default). If the old system does not have canonical IDs, each match may instead
contain `video_id`, `start_seconds`, and `end_seconds`; the command resolves the
timestamp to a canonical segment. It validates all references and ranks before
searching, writes neutral overlap statistics, and creates
`batch-comparison/blind_review.csv`. The worksheet hides which system produced
set A or B so a reviewer can judge action, object, tool, and overall relevance
without seeing the method name.

Re-running the command cannot erase a worksheet that already contains human
labels or notes. New artifacts receive a `.generated` filename so the completed
review and its matching hidden ranking key stay together.

After filling the judgment columns with `yes` or `no`, run:

```bash
./scripts/run_local_pipeline.sh score-review
```

This calculates original-versus-new Precision@3, Success@3, per-query
wins/ties/losses, and a paired bootstrap confidence interval. Blank cells stay
missing; they are never counted as negative judgments.

The exact input contract is in
`data_contracts/original_rankings.schema.json`.

## What has been accomplished

- The private nested JSONL is strictly validated and parsed completely.
- Invalid intervals and duplicate temporal segments are handled explicitly.
- Stable canonical clip IDs and a reproducible build manifest are generated.
- Action, object, direct tool, tool/supply inventory context, spatial scope,
  VerbNet, and FrameNet features are extracted from real records.
- Search works with lexical, structured, and hybrid ranking.
- Short commands support top-k search and optional one-result-per-video output.
- One-query comparison no longer makes circular improvement claims.
- Batch comparison accepts real original rankings and produces a blind review
  worksheet.
- Original results can be supplied as canonical IDs or source video/timestamp
  intervals, and batch queries reuse one loaded lexical/structured index.
- Standalone commands rebuild when the source, extraction code, spaCy/model
  versions, scorer version, or generated artifact hashes no longer match.
- Completed blind worksheets can be scored without revealing A/B assignments
  during labeling, and reruns preserve existing human work.
- The direct-title and exact-phrase controlled benchmark reports category
  results, score coverage, tie-aware metrics, and paired 95% confidence
  intervals.
- Retrieval artifacts record the exact input hashes, model, scorer version,
  field policy, and hybrid weight needed to reproduce a run.
- The automated suite currently has 59 passing tests and Ruff is clean.

## What I need next

The most important missing data is not another copy of the same clips. It is the
actual old retrieval output for a set of queries.

I should request:

1. A query or step ID and its text.
2. The original top three clip IDs in their exact rank order, or enough video
   ID/timestamp information to map them to canonical IDs.
3. At least 50–100 queries across several categories.
4. Permission to inspect the pooled timestamped clips and record blinded human
   relevance judgments.
5. A larger export in the same IndexedVideo format for scale testing.
6. More descriptions, goals, captions, or transcripts if available. Only 35.0%
   of the canonical sample has a clip description, so the current automatic
   benchmark covers the richer part of the data.

The next technical work is to label the blind comparison, study the structured
method’s failures, improve action parsing and object/tool alignment, tune any
hybrid weight on separate development videos, and evaluate once on held-out
videos. I should not tune the score on the same 462 queries used for the final
reported result.

## Smaller commands

```bash
./scripts/run_local_pipeline.sh test
./scripts/run_local_pipeline.sh build
./scripts/run_local_pipeline.sh benchmark
./scripts/run_local_pipeline.sh review
./scripts/run_local_pipeline.sh score-review
./scripts/run_local_pipeline.sh --help
```

`DATA_COMPLETION_STATUS.md` records the remaining data work, and
`LOCAL_VERIFICATION.md` records the exact local verification run.
