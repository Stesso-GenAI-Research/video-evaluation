# Action Semantics for Instructional Video Search

## What this project is trying to do

This project asks a simple search question:

> If a project step says “remove the old faucet,” can we find clips that show
> that exact action on that exact object?

Ordinary text search is good at finding clips about the same topic. It may still
confuse opposite or nearby actions, such as *remove a faucet* and *install a
faucet*. I am testing whether an explicit representation of each clip's action,
object, tool, and supply can make the result list more precise.

The repository compares three methods:

- **Lexical:** TF-IDF word matching, which is the ordinary text baseline.
- **Structured:** matching parsed action–object–tool/supply records.
- **Hybrid:** a 50/50 combination of lexical and structured scores.

## What currently works

| Part | Current state |
|---|---|
| Parse the private JSONL | Working on the supplied nested export |
| Build a clean clip index | Working; canonical timestamped clips are deduplicated |
| Extract action semantics | Working, with measured coverage and unmeasured precision |
| Search the clips | Working with lexical, structured, and hybrid ranking |
| Compare two result lists | Working for one query; batch mode is ready for supplied old results |
| Make a blinded review sheet | Working and protected from accidental overwrite |
| Evaluate already-judged A/B pairs | Working; frozen direct scoring and step-clustered intervals are ready for W25 |
| Build controlled hard-negative development pairs | Working; metadata-derived labels are diagnostic only |
| Build large score-independent contrast pairs | Working |
| Inventory pairwise scorer disagreements | Working; descriptive case table with no tuning |
| Run an automatic benchmark | Working as a development test |
| Prove the new search is better | **Not complete; the real W25 preference export is still needed** |

This is a local research pipeline, not a deployed website or search API. It
does, however, provide the functional search and comparison tools needed to run
the research experiment.

## Quick start

Python 3.11, 3.12, or 3.13 is required. The first command installs the project's
Python packages plus the spaCy and NLTK resources, so it needs an internet
connection:

```bash
./scripts/run_local_pipeline.sh setup
```

Run the automated checks, rebuild the index, and run the development benchmark:

```bash
./scripts/run_local_pipeline.sh all
```

Run a real search over the indexed clips:

```bash
./scripts/run_local_pipeline.sh search "remove old faucet"
```

The default returns the top three hybrid results. For example, this runs the
ordinary text baseline and limits the result list to one clip per video:

```bash
./scripts/run_local_pipeline.sh search "paint wall" \
  --method lexical --top-k 5 --max-per-video 1
```

Generated files are placed under
`project1_outputs/indexed-video-sample/`. The `all` command does not run a human
review or invent old search results; those steps require real experiment input.

## Current results and next experiment

The current development evidence favors lexical TF-IDF over structured and
hybrid scoring. The primary frozen pairwise-preference experiment is ready but
still requires the authentic W25 steps and already-observed A/B judgments.
Constructed development pairs verify the full workflow but are not human
preference evidence.

Current counts, result tables, exact commands, output paths, limitations, and
the Stesso meeting update are maintained in the
[project status, experiments, and execution guide](docs/stesso-project-update.md).

## Documentation

- [Project status, experiments, and execution guide](docs/stesso-project-update.md)
- [How the system works](docs/how-it-works.md)
- [Running the pipeline and finding its outputs](docs/running-the-pipeline.md)
- [Pairwise preference evaluation (Project 1, Step 1)](docs/pairwise-preference-evaluation.md)
- [Experiment methodology and interpretation](docs/experiments-and-results.md)
- [Current verification record](docs/verification.md)
- [Input data contracts](data_contracts/README.md)
