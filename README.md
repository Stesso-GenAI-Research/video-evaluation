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
| Parse the private JSONL | Working on all 250 supplied videos |
| Build a clean clip index | Working; 1,663 searchable timestamped clips |
| Extract action semantics | Working, with measured coverage and unmeasured precision |
| Search the clips | Working with lexical, structured, and hybrid ranking |
| Compare two result lists | Working for one query; batch mode is ready for supplied old results |
| Make a blinded review sheet | Working and protected from accidental overwrite |
| Evaluate already-judged A/B pairs | Working; frozen direct scoring and step-clustered intervals are ready for W25 |
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

## What we have found so far

The automatic development benchmark uses clip names as queries and their paired
descriptions, goals, tools, and supplies as candidate evidence. It is useful for
debugging, but it is not a substitute for human relevance labels.

| Whole-corpus method | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| Lexical TF-IDF | 63.2% | 85.5% | 0.749 |
| Structured action | 26.4% | 36.3% | 0.335 |
| 50/50 hybrid | 50.6% | 61.5% | 0.577 |

The honest result is that the current structured method does **not** beat
TF-IDF over the whole corpus. It misses some actions and produces many tied
scores. Within one source video, lexical and hybrid search both reached 72.7%
Hit@1, but the uncertainty interval is too wide to claim that they are equal.
This suggests that action features may be more useful for reranking clips that
are already about the same topic than for searching the entire corpus alone.

## What comes next

The immediate milestone is the frozen pairwise-preference evaluation on the
real W25 export. The command and contracts are ready; the remaining external
input is the indexed videos, project steps, and already-observed A/B judgments.
Synthetic fixtures verify the machinery but are not research evidence.

## Documentation

- [How the system works](docs/how-it-works.md)
- [Running the pipeline and finding its outputs](docs/running-the-pipeline.md)
- [Pairwise preference evaluation (Project 1, Step 1)](docs/pairwise-preference-evaluation.md)
- [Experiments, findings, missing data, and next steps](docs/experiments-and-results.md)
- [Current verification record](docs/verification.md)
- [Input data contracts](data_contracts/README.md)
