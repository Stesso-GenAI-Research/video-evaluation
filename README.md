# Action Semantics for Indexed Video Search

## What this project is supposed to do

The immediate goal is to build a working search tool for indexed instructional video clips.

A user enters an action-based query such as:

```text
remove old faucet
```

The program searches all indexed clips and returns the three closest matches. It pays attention to the action and object, rather than only looking for videos about the same broad topic.

For this query, the current system returns clips such as:

```text
1. Remove Faucet Handle
2. Remove Faucet Stem
3. Remove Corroded Tee Fitting
```

The first two results match both the action `remove` and the object `faucet`. The third matches the action but not the object, so it receives a lower score.

That is the first deliverable: use action semantics to search the indexed clips and return useful top results.

The second deliverable compares two result sets for the same step:

```text
Original top 3          Action-semantic top 3
        \                     /
         compare overlap, action, object, tool, and text scores
```

## Why `indexed-videos-250.jsonl` is enough

I originally treated the nested `clips` list as if it were missing a separate step table. That was too strict.

Each object inside `clips` is already an indexed, timestamped instructional step. It contains:

- `name`: the short action or step name
- `description`: a longer explanation when available
- `goal`: the result of the step when available
- `tools`: tool metadata
- `supplies`: material metadata
- `start` and `end`: the location of the clip inside the video

The parent video also supplies its ID, title, summary, goal, category, URL, and YouTube ID.

This is enough to create a searchable corpus. The current sample contains 250 videos and 1,703 indexed clips. A larger file in the same format will let the same code search a larger corpus.

Separate project steps and pairwise human labels could still be useful for a later formal accuracy study, but they are not required to make the top-three search system run.

## Run a search

First prepare and index the sample:

```bash
./scripts/run_local_pipeline.sh all
```

Then search it:

```bash
./scripts/run_local_pipeline.sh search "remove old faucet"
```

Another example is:

```bash
./scripts/run_local_pipeline.sh search "assess scum buildup"
```

The search command prints the top three matches and saves a JSON result under:

```text
project1_outputs/indexed-video-sample/search_<query>.json
```

Each result includes:

- clip and video IDs
- clip and video titles
- YouTube URL
- start and end timestamps
- total structured score
- action, object, tool, VerbNet, FrameNet, and taxonomy match scores

## Run a comparison

Compare a normal text-search top three with the new action-semantic top three:

```bash
./scripts/run_local_pipeline.sh compare "remove old faucet"
```

The command saves:

```text
project1_outputs/indexed-video-sample/comparison_remove_old_faucet.json
```

The comparison reports:

- the original and new ranked clips
- clips shared by both sets
- clips found only by one method
- mean text score for each set
- mean structured score for each set
- action, object, tool, VerbNet, FrameNet, and taxonomy averages

For `remove old faucet`, the default text baseline retrieved clips containing words such as `remove` and `old`, including an unrelated old control and old filter. The action-semantic results retrieved `Remove Faucet Handle` and `Remove Faucet Stem`. The mean structured score increased from about `0.47` to `0.77`.

By default, “original” means a TF-IDF text-search baseline because the current JSONL does not mark results with an `original_matches` field. If real original clip IDs are available for a step, compare those exact results with:

```bash
ORIGINAL_CLIP_IDS="clip-id-1,clip-id-2,clip-id-3" \
  ./scripts/run_local_pipeline.sh compare "remove old faucet"
```

This makes the comparison code usable now and ready for actual original result IDs later.

## How the search works

The pipeline first reads all 1,703 clips and extracts structured information from their text.

For example:

```text
Tighten the loose pipe connection with a wrench.
```

becomes approximately:

```text
action: tighten
object: pipe connection
tool: wrench
```

When a user enters a search query, the same parser extracts its action and object. The program compares the query with every indexed clip.

The score uses six signals:

1. Exact action match
2. Object match
3. Tool match
4. VerbNet action-class match
5. FrameNet frame match
6. DIY action-taxonomy match

The clips are sorted by the combined score, and the top three are returned.

The current search is structured-only. Dense embeddings and a hybrid search can be added later, but they are not needed for the action-semantic search to work.

## What has been accomplished

The repository now has a complete local path that:

1. Reads the nested IndexedVideo JSONL format.
2. Creates stable local clip IDs.
3. Preserves video URLs and clip timestamps.
4. Extracts actions, objects, tools, and materials.
5. Maps actions through VerbNet and FrameNet.
6. Builds an early DIY action taxonomy.
7. Creates a searchable structured index.
8. Returns the top three matches for a new action query.
9. Compares an original/text result set with the new result set.
10. Generates quality reports and a human-review worksheet.
11. Verifies the output files.

The code currently has 17 passing tests, and Ruff reports no code-quality errors.

## Current sample results

After improving the parser for short title-style instructions, the current run produced:

- 250 videos
- 1,703 indexed clips
- 9,246 extracted action records
- 1,023 unique action lemmas
- actions found in 1,335 clips, or 78.4%
- 368 clips with no extracted action
- objects found in 64.6% of action records
- directly connected tools found in 7.6%
- VerbNet mappings for 70.9% of action lemmas
- FrameNet mappings for 69.2%
- 670 recurring actions placed into 80 early DIY groups

Tool matching is still the weakest signal. Many clips list tools in metadata without using a sentence such as “cut the board with a saw.” The pipeline keeps metadata tools as a separate fallback so they can help search without being confused with a certain grammatical connection.

## Quality review

The file below contains 60 extracted examples for manual checking:

```text
project1_outputs/indexed-video-sample/quality/manual_review_sample.csv
```

For each row, enter `yes` or `no` in:

- `action_correct`
- `object_correct`
- `tool_correct`

Then run:

```bash
./scripts/run_local_pipeline.sh review
```

This produces `manual_review_results.json` with human-reviewed precision. This review measures how reliable the parser is; it does not prevent us from running and inspecting searches now.

## What should happen next

The next development work is:

1. Run a set of realistic searches and inspect the top three results.
2. Label the 60-row quality worksheet.
3. Fix the most common action, object, and tool errors.
4. Run the same pipeline on a larger IndexedVideo export.
5. Feed actual original match IDs into the working comparison command.
6. Add a repeatable benchmark using clip names as queries and their aligned clips as known matches.
7. Add an embedding baseline and test a hybrid ranking.

The most useful new data is simply more records in the same format. It is especially helpful if more clips contain descriptions, goals, captions, or transcripts. Real database clip IDs would also be helpful, but the current code can generate stable local IDs from the video ID and clip position.

## Other commands

```bash
./scripts/run_local_pipeline.sh setup   # install the project
./scripts/run_local_pipeline.sh test    # run tests and code checks
./scripts/run_local_pipeline.sh sample  # rebuild the structured index
./scripts/run_local_pipeline.sh search "install a thermostat"
./scripts/run_local_pipeline.sh compare "install a thermostat"
./scripts/run_local_pipeline.sh review  # summarize manual labels
./scripts/run_local_pipeline.sh --help
```

The main generated files are under:

```text
project1_outputs/indexed-video-sample/
```

The input schemas are in `data_contracts/`, the detailed data status is in `DATA_COMPLETION_STATUS.md`, and the local test record is in `LOCAL_VERIFICATION.md`.
