# Action Semantics for Instructional Video Search

## The point of this project

This project asks one main question:

> Can a video search system find a better instructional clip by understanding the exact action being performed?

Imagine that a project step says:

> Tighten the loose pipe connection with a wrench.

A normal text-search or embedding system may return any clip about plumbing. It could return a clip about painting a pipe, replacing a pipe, or inspecting a leak. Those clips are related to the same topic, but they do not show the requested action.

I want the search system to notice three important details:

- The action is `tighten`.
- The object is a `pipe connection`.
- The tool is a `wrench`.

The research question is whether matching these details improves search results compared with matching general text meaning alone.

## The experiment I eventually want to run

For each real project step, I will give the system two candidate video clips. A person has already decided which clip is the better match.

I will test three search methods:

1. Dense matching compares the general meaning of the step and clip text using embedding vectors.
2. Structured matching compares actions, objects, tools, and related action groups.
3. Hybrid matching combines dense and structured matching.

Each method chooses one of the two clips. I then check whether it chose the same clip as the human reviewer.

The main result will be pairwise accuracy: the percentage of comparisons where the system agrees with the human.

This is the actual research experiment. Everything completed so far prepares the data and features needed for it.

## What I have built so far

The available private sample is `indexed-videos-250.jsonl`. It contains 250 videos and 1,703 shorter clips.

I built an automatic pipeline that:

1. Converts the nested video file into one record per clip.
2. Reads clip titles, descriptions, and goals.
3. Finds possible action words such as `cut`, `remove`, `install`, and `paint`.
4. Tries to find the object receiving the action.
5. Tries to find a tool directly connected to the action.
6. Keeps tools listed in metadata as a separate, less certain fallback.
7. Looks up actions in VerbNet and FrameNet, which are existing English-language resources.
8. Groups similar action words into an early DIY action taxonomy.
9. Checks that the expected output files exist and contain valid rows.
10. Creates a small worksheet for a person to check whether the extraction is correct.

The code currently has 15 passing tests. Setup, testing, sample analysis, and verification can all be run from one Bash script.

## What I found

The pipeline processed all 1,703 clips.

- It produced 8,823 possible action records.
- It found at least one action in 1,100 clips, or 64.6%.
- It found no action in 603 clips.
- It found an object for 63.1% of the action records.
- It found a tool directly connected to an action for only 3.5%.
- VerbNet contained a mapping for 71.1% of the action words.
- FrameNet contained a mapping for 69.4%.
- The early DIY taxonomy contains 660 recurring actions in 80 groups.

The clearest finding is that tool information is difficult to connect to actions. Many clips list tools in metadata, but the clip text does not say something direct like “cut the board with a saw.” I now preserve that metadata as a fallback, but I keep it separate because a listed tool may not belong to every action in the clip.

These numbers describe extraction coverage. They do not tell me whether the extraction is correct, and they do not prove that this method improves search.

## Why there is a manual review worksheet

A parser can produce an answer that looks reasonable but is wrong. For example, it may treat the wrong noun as an object or treat a phrase after the word “by” as a tool.

The pipeline creates:

```text
project1_outputs/indexed-video-sample/quality/manual_review_sample.csv
```

This file contains 60 examples. For each row, I need to enter `yes` or `no` in:

- `action_correct`
- `object_correct`
- `tool_correct`

I can leave a cell blank when that part does not apply. After labeling the worksheet, I run:

```bash
./scripts/run_local_pipeline.sh review
```

The script calculates human-reviewed precision and writes:

```text
project1_outputs/indexed-video-sample/quality/manual_review_results.json
```

This tells me how often the extracted actions, objects, and tools are actually correct in the reviewed sample.

## What I have not proven

I have not yet shown that structured action matching improves video search.

The current sample contains video clips, but it does not contain:

- Real project steps to use as search queries.
- Human decisions about which clips best match those steps.
- Dense embedding vectors for both steps and clips.

Without those three connected pieces, there is no fair search experiment to run. Creating fake queries, labels, or vectors would produce a meaningless accuracy number.

### 1. Project steps

Each step should include:

- `step_id`
- title and description
- tools, materials, and techniques when available
- dense embeddings

These steps are the search queries.

### 2. Human clip comparisons

Each comparison should include:

- `comparison_id`
- `step_id`
- `clip_a_id`
- `clip_b_id`
- `winner_clip_id`
- tie or uncertainty information, if it exists

These comparisons provide the correct answers for the experiment.

### 3. The referenced clips

Each clip should include:

- `clip_id`
- title, description, and summary
- transcript or captions when available
- Gemini metadata
- dense embeddings

The same IDs must be used in all three files so the steps, candidates, and human answers can be joined correctly.

I also need documentation for the embedding model: its name, vector size, source text, and whether the step and clip embeddings were generated in the same way.

## What happens next

The next work should happen in this order:

1. Label the 60-row review worksheet.
2. Calculate action, object, and tool precision.
3. Inspect the 603 clips where no action was found.
4. Fix the most common extraction mistakes.
5. Obtain the step, comparison, clip, and embedding exports.
6. Validate that all IDs and vectors connect correctly.
7. Run dense, structured, and hybrid matching.
8. Compare each method with the human choices.
9. Report which method performs best and how certain the result is.

At that point, I can answer the real research question: does understanding the exact action help retrieve a better instructional video?

## Running the project

The normal local command is:

```bash
./scripts/run_local_pipeline.sh all
```

It creates the Python environment, installs dependencies, runs the tests, analyzes the sample, performs the quality checks, and verifies the output files.

Other commands are:

```bash
./scripts/run_local_pipeline.sh setup   # install the project
./scripts/run_local_pipeline.sh test    # run tests and code checks
./scripts/run_local_pipeline.sh sample  # analyze the 250-video sample
./scripts/run_local_pipeline.sh review  # summarize human review labels
./scripts/run_local_pipeline.sh --help
```

The main output files are:

- `sample_analysis_report.json`: overall sample results
- `month1/action_object_tool_triples.csv`: extracted actions, objects, and tools
- `month2/diy_actionnet_v1.jsonl`: early DIY action groups
- `quality/extraction_quality_summary.json`: coverage statistics
- `quality/manual_review_sample.csv`: examples to label manually
- `structured_analysis_verification_report.json`: output validation results

The complete Month 3 experiment can be run later with:

```bash
CLIPS_JSONL=/secure/path/clips.jsonl \
STEPS_JSONL=/secure/path/steps.jsonl \
PAIRWISE_JSONL=/secure/path/pairwise.jsonl \
PROJECT_OUTPUT_DIR=/secure/path/project1_outputs \
  ./scripts/run_local_pipeline.sh full
```

The input schemas are in `data_contracts/`. The current data status is in `DATA_COMPLETION_STATUS.md`, and the local test record is in `LOCAL_VERIFICATION.md`.
