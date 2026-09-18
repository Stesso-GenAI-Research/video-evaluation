# Current verification record

Verified September 18, 2026 on macOS 27.0 with Python 3.13.15.

Current data counts and experiment results are maintained in the
[project status, experiments, and execution guide](stesso-project-update.md).
This page records verification scope only.

## Automated checks

Run:

```bash
./scripts/run_local_pipeline.sh test
```

Latest result:

```text
93 tests passed
All Ruff checks passed
Python source and tests compiled
Bash syntax check passed
CLI help rendered successfully
```

The suite covers:

- nested input parsing, invalid interval quarantine, duplicate merging, and
  canonical clip IDs;
- action extraction and lexical, structured, and hybrid scoring;
- score normalization and configuration guardrails;
- single-query and batch ranking comparison;
- blind assignment and worksheet overwrite protection;
- timestamp and exact-ID resolution, including missing and ambiguous cases;
- review scoring, label coverage, and paired bootstrap intervals;
- field-held-out benchmark controls and tie-aware metrics;
- index source, code, model, configuration, and artifact freshness;
- frozen pairwise scoring, winner mapping, validation, and provenance retention;
- step-clustered bootstrap sampling and fixed-seed reproducibility;
- constructed hard-negative and score-independent contrast generation;
- pair diagnostic counts and case-table generation; and
- end-to-end command smoke tests.

## Real sample checks

The build was run against the real `indexed-videos-250.jsonl` sample. It
successfully produced the canonical clip index, rejected-row inventory, action
records, quality worksheet, retrieval benchmark inputs, and index manifest.

`index_manifest.json` records the source SHA-256 hash, generated artifact hashes,
spaCy model, extraction and schema versions, scorer version, and candidate-field
policies. Freshness checks verify the source, relevant code, model,
configuration, and output artifacts before reuse.

The extraction review at
`project1_outputs/indexed-video-sample/quality/manual_review_sample.csv` remains
unlabeled. Extraction coverage is measured; extraction precision is not.

## Functional search checks

The following paths were exercised against the canonical sample:

```bash
./scripts/run_local_pipeline.sh search "remove old faucet" --method lexical
./scripts/run_local_pipeline.sh search "remove old faucet" --method structured
./scripts/run_local_pipeline.sh search "remove old faucet" --method hybrid
./scripts/run_local_pipeline.sh search "paint wall" --method hybrid --max-per-video 1
./scripts/run_local_pipeline.sh search "paint wall with primer" --method hybrid
./scripts/run_local_pipeline.sh search "tighten screw with screwdriver" \
  --method hybrid --max-per-video 1
```

These checks reached lexical, action/object, tool/supply, short-command fallback,
per-video diversity, and hybrid lexical-fallback behavior. They verify that the
paths execute and return indexed clips; they are not an accuracy sample.

## Ranking-comparison checks

Automated tests verify that `compare-batch`:

- requires unique step IDs and contiguous original ranks;
- accepts canonical IDs or video/start/end references;
- resolves timestamps with the configured tolerance;
- rejects missing and ambiguous references before search;
- preserves supplied historical rank order;
- reuses loaded indexes across a batch;
- writes neutral overlap statistics and a deterministic blinded worksheet;
- hides original/challenger method identity; and
- protects entered labels by writing regenerated files with a suffix.

Review-scoring tests cover worksheet identity, label parsing, rank integrity,
hidden A/B assignment, incomplete-label coverage, Precision@*k*, Success@*k*,
wins/ties/losses, and paired intervals.

No real old-versus-new worksheet exists because historical rankings have not
been supplied. Test fixtures verify the machinery only.

## Pairwise-preference checks

Both current constructed corpora were generated, evaluated, and analyzed from
the canonical sample with seed 42. Verification confirmed:

- every generated reference resolves to the intended canonical clip;
- no input row is silently dropped;
- A and B winner positions map correctly;
- lexical, structured, and hybrid scores and credits are recorded independently;
- exact score ties receive credit `0.5`;
- frozen hybrid alpha remains 0.5;
- provenance and confidence survive input to output;
- bootstrap draws preserve complete `step_id` clusters;
- fixed-seed intervals are reproducible; and
- diagnostic outputs match the pair-level score records.

The generated manifests explicitly record `human_judgments: false` and
`research_evidence: false`. Current corpus sizes, resolution totals, agreement
rates, intervals, and diagnostic counts are reported only in the
[canonical project guide](stesso-project-update.md).

## Open manual and external checks

The following cannot be completed automatically:

1. Label the 60-row extraction-quality worksheet to estimate parser precision.
2. Supply and independently label historical old-versus-new rankings for the
   blinded retrieval comparison.
3. Supply the authentic W25 steps and already-observed A/B judgments to obtain
   the primary frozen Step 1 result.

Constructed labels and automated smoke tests do not replace these checks.
