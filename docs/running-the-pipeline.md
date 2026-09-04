# Running the local pipeline

The Bash script in `scripts/run_local_pipeline.sh` is the normal entry point.
It supplies the long file paths for the Python CLI and keeps the common workflow
in one place.

Run every command below from the repository root.

## Requirements and one-time setup

Use Python 3.11, 3.12, or 3.13. Then run:

```bash
./scripts/run_local_pipeline.sh setup
```

This command creates `.venv`, installs the project and development tools,
downloads the `en_core_web_sm` spaCy model, and downloads the NLTK WordNet,
VerbNet, and FrameNet resources. The downloads require an internet connection.
Later search and experiment runs are local.

## The normal complete run

```bash
./scripts/run_local_pipeline.sh all
```

`all` performs these steps in order:

1. Compile the source and tests.
2. Run the test suite and Ruff checks.
3. Check the Bash syntax and Python command-line help.
4. Rebuild the canonical clip and action-semantic index from the JSONL.
5. Run the automatic lexical, structured, and hybrid benchmark.

It creates the environment first only when `.venv` is missing. It does **not**
run example queries, compare against supervisor results, or fill either manual
review worksheet. Those activities require a query or a human decision.

## Command reference

| Command | What it does |
|---|---|
| `setup` | Create the environment and install all dependencies. |
| `test` | Run compilation, tests, Ruff, Bash syntax, and CLI checks. |
| `build` | Parse the private JSONL and rebuild all index artifacts. |
| `sample` | Older name for `build`; both run the same code. |
| `search` | Search the canonical clips with one of the three methods. |
| `compare` | Describe differences between two rankings for one query. |
| `compare-batch` | Compare many supplied old rankings and make a blind worksheet. |
| `evaluate-pairs` | Directly score already-judged A/B pairs for frozen Step 1. |
| `score-review` | Score a completed old-versus-new worksheet. |
| `benchmark` | Run the automatic field-pair development benchmark. |
| `review` | Summarize labels in the extraction-quality worksheet. |
| `all` | Run `test`, `build`, and `benchmark`. |
| `--help` | Print the Bash command summary. |

For example:

```bash
./scripts/run_local_pipeline.sh test
./scripts/run_local_pipeline.sh build
./scripts/run_local_pipeline.sh benchmark
./scripts/run_local_pipeline.sh --help
```

The script stops at the first failed step and returns a nonzero exit code. This
makes it suitable for both interactive work and automated checks.

## Searching clips

The smallest search command is:

```bash
./scripts/run_local_pipeline.sh search "remove old faucet"
```

It uses hybrid search and asks for three results. Search flags written after the
query are passed to the Python CLI:

```bash
./scripts/run_local_pipeline.sh search "paint wall" \
  --method lexical \
  --top-k 5 \
  --max-per-video 1
```

Useful options are:

| Option | Meaning |
|---|---|
| `--method lexical` | Use only TF-IDF word matching. |
| `--method structured` | Use only parsed action-semantic matching. |
| `--method hybrid` | Combine lexical and structured scores; this is the default. |
| `--top-k N` | Request up to *N* positive-score results. |
| `--max-per-video N` | Limit how many results may come from one source video. |
| `--hybrid-alpha X` | Give fraction *X* to lexical search and `1-X` to structured search. |

`--hybrid-alpha` must be from 0 to 1 and defaults to 0.5. A query can return
fewer than *k* clips when fewer than *k* receive a positive score. The program
does not add zero-score filler results.

Every run prints its result information and writes a JSON file named from the
query plus a short configuration hash. Changing a flag therefore does not
silently overwrite a different version of the same query.

## Comparing rankings

For a quick machinery check, compare the generated TF-IDF baseline with the
hybrid result:

```bash
./scripts/run_local_pipeline.sh compare "remove old faucet"
```

This reports overlap, Jaccard similarity, set differences, and rank changes. It
does not select a winner. Without explicit original IDs, the reference is a
newly generated lexical baseline, not the private system's historical result.

Explicit original clip IDs may be supplied after the query:

```bash
./scripts/run_local_pipeline.sh compare "remove old faucet" \
  --original-clip-id indexed-video-1451721-segment-94p616-118p3 \
  --original-clip-id indexed-video-1451721-segment-61p383-79p466
```

For the real multi-query experiment, use a JSONL file that follows the
[original ranking contract](../data_contracts/README.md):

```bash
./scripts/run_local_pipeline.sh compare-batch path/to/original_rankings.jsonl
```

This validates the full input before searching, loads the indexes once for the
batch, generates challenger results, and writes a pooled worksheet in a hidden
A/B order. The detailed research procedure is in
[Experiments and results](experiments-and-results.md#the-real-old-versus-new-experiment).

## Evaluating already-judged pairs

The Step 1 preference evaluation is separate from `compare-batch`: it accepts
an A/B decision that already exists and scores those two clips directly. It
does not retrieve a challenger list. Run it against the nested W25 export with:

```bash
./scripts/run_local_pipeline.sh evaluate-pairs \
  --index data/indexed-videos-w25.jsonl \
  --steps data/steps-w25.jsonl \
  --pairs data/pairwise-w25.jsonl \
  --output project1_outputs/w25/step1 \
  --seed 42
```

The command writes `pair_scores.jsonl`, `summary.json`, `summary.md`,
`validation_report.json`, and `manifest.json`. Its scoring configuration is
frozen; only the bootstrap seed and number of bootstrap iterations are run
controls. See the dedicated
[pairwise-preference guide](pairwise-preference-evaluation.md) for the current
contracts, built-index form, resolution policy, and interpretation.

## Completing a blinded review

After `compare-batch`, open:

```text
project1_outputs/indexed-video-sample/batch-comparison/blind_review.csv
```

For each pooled clip, enter `yes` or `no` in these columns:

- `overall_relevant`: the clip is a useful answer to the query;
- `action`: the clip performs the requested action;
- `object`: it acts on the requested object;
- `tool`: it uses the requested tool, when the query asks for one.

Use `notes` for uncertainty or a short reason. Leave a field blank when it
cannot be judged. Blank values remain missing and are not converted into “no.”

Then run:

```bash
./scripts/run_local_pipeline.sh score-review
```

The scorer checks that the worksheet still matches its hidden A/B assignment.
It reports label coverage, Precision@*k*, Success@*k*, wins/ties/losses, and a
paired bootstrap confidence interval. Only steps with complete labels for a
dimension contribute to that dimension's comparison.

If `compare-batch` is rerun after human labels or notes have been entered, it
does not overwrite the work. The new files receive a `.generated` suffix.

## Reviewing extraction quality

The index build creates a separate 60-row worksheet at:

```text
project1_outputs/indexed-video-sample/quality/manual_review_sample.csv
```

This worksheet checks whether the parser extracted the action, object, and
direct tool correctly. Fill `action_correct`, `object_correct`, and
`tool_correct`, then run:

```bash
./scripts/run_local_pipeline.sh review
```

This review measures parser precision. It is different from the blinded search
review, which measures whether entire result clips answer a query.

## Output directory map

The main build directory is:

```text
project1_outputs/indexed-video-sample/
├── input/
│   ├── indexed_video_clips.jsonl       canonical searchable clips
│   ├── indexed_video_profile.json      source coverage and quality counts
│   └── rejected_clips.jsonl            invalid source annotations
├── month1/                              extracted triples and VerbNet data
├── month2/                              FrameNet, role, and taxonomy diagnostics
├── quality/
│   └── manual_review_sample.csv         extraction review worksheet
├── benchmark/
│   ├── benchmark_summary.json           aggregate development results
│   └── benchmark_queries.csv            one row per benchmark query
├── batch-comparison/                     created after compare-batch
├── index_manifest.json                   input, code, model, and output hashes
└── search_*.json / comparison_*.json     individual runs
```

The `month1` and `month2` directory names are historical development names.
They are both part of the current build; the user does not run separate months.
The private JSONL and generated output directory are ignored by Git; source code
and documentation are the versioned parts of the project.

## Index freshness

`search`, `compare`, `compare-batch`, and `benchmark` check the existing index
before using it. They rebuild when the source JSONL, extraction code, spaCy
model, scoring version, or generated artifact hashes no longer match. The
manifest's Git commit is the commit that existed when the index was built; the
hash checks are what determine whether the current artifacts are safe to reuse.

## Changing input or output locations

The Bash wrapper reads these optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SAMPLE_JSONL` | `indexed-videos-250.jsonl` | Nested IndexedVideo export to parse. |
| `SAMPLE_OUTPUT_DIR` | `project1_outputs/indexed-video-sample` | Main artifact directory. |
| `PROJECT_OUTPUT_DIR` | `project1_outputs` | Parent output directory. |
| `VENV_DIR` | `.venv` | Python virtual environment. |

Example:

```bash
SAMPLE_JSONL=/path/to/indexed-videos-larger.jsonl \
SAMPLE_OUTPUT_DIR=project1_outputs/indexed-video-larger \
  ./scripts/run_local_pipeline.sh all
```

## Common problems

- **The input file is missing:** place the private export at the repository
  root or set `SAMPLE_JSONL` to its path.
- **Python is unsupported:** install Python 3.11–3.13. The setup script checks
  for those versions in that order.
- **An NLTK or spaCy resource is missing:** rerun `setup` while connected to the
  internet.
- **The index rebuilds unexpectedly:** this is normally a freshness check doing
  its job. Read `index_manifest.json` to see the recorded input and versions.
- **Hybrid search reports a parsing warning:** the query had no usable action
  record, so hybrid search returned lexical results instead of pretending it had
  structured evidence.
- **No batch comparison directory exists:** the real supervisor ranking file
  has not been passed to `compare-batch` yet.
