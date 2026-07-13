# Local verification notes

The automated pipeline was checked locally with Python 3.13. The machine's default Python was 3.14, which is outside the project's declared Python range, so the virtual environment was created with Python 3.13.

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -m nltk.downloader verbnet wordnet omw-1.4 framenet_v17

.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/action-semantics --help
```

The test suite passed locally with 17 tests. Compilation and Ruff also passed. The installed CLI includes working `search-indexed-clips` and `compare-indexed-clips` commands.

The following command built the structured index from the real private sample:

```bash
.venv/bin/action-semantics run-indexed-video-analysis \
  --indexed-videos-jsonl indexed-videos-250.jsonl \
  --output-dir /tmp/action-semantics-sample-results \
  --min-taxonomy-support 2

.venv/bin/action-semantics verify-structured-outputs \
  --output-dir /tmp/action-semantics-sample-results
```

The run flattened 250 videos into 1,703 clip records. It generated 9,246 action triples from 1,023 unique action lemmas; 64.6% of triples had an extracted object and 7.6% had a dependency-linked tool. The taxonomy contains 670 recurring actions in 80 clusters. All required JSONL artifacts were nonempty and had their required fields.

The July 2026 quality audit found actions in 1,335 clips (78.4%), record-level tool context for 87.1% of triples, VerbNet mappings for 70.9% of action lemmas, and FrameNet mappings for 69.2%. It also generated 60 deterministic manual-review rows across four error-analysis groups. Record-level tool context is reported separately from direct tool extraction because availability is not the same as correctness.

The query `remove old faucet` returned `Remove Faucet Handle` and `Remove Faucet Stem` as the first two results. The query `assess scum buildup` returned the exact source clip first, followed by two other `assess` actions. These are working retrieval examples, not human-reviewed accuracy measurements.

The functional comparison for `remove old faucet` compared a TF-IDF text top three with the action-semantic top three. The sets had no overlap. Mean structured score increased from about 0.47 for the text results to 0.77 for the action results, while the text baseline retained the higher mean lexical score, as expected.
