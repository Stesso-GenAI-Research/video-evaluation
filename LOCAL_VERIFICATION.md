# Local verification record

Verified July 13, 2026 on macOS with Python 3.13.14.

## Commands

```bash
./scripts/run_local_pipeline.sh test
./scripts/run_local_pipeline.sh build
./scripts/run_local_pipeline.sh benchmark

./scripts/run_local_pipeline.sh search "remove old faucet" --method lexical
./scripts/run_local_pipeline.sh search "remove old faucet" --method structured
./scripts/run_local_pipeline.sh search "remove old faucet" --method hybrid
./scripts/run_local_pipeline.sh search "paint wall" --method hybrid --max-per-video 1
./scripts/run_local_pipeline.sh search "tighten screw with screwdriver" \
  --method hybrid --max-per-video 1
```

The test command runs compilation, pytest, Ruff, CLI help, and the Bash syntax
check. The final result was:

```text
51 passed
All Ruff checks passed
CLI help rendered successfully
Bash syntax check passed
```

## Ingest and index checks

The real private sample was parsed without synthetic records:

```text
source videos:                    250
raw nested clip rows:           1,703
valid source rows:              1,700
invalid rows quarantined:           3
duplicate valid rows merged:       37
canonical searchable segments: 1,663
```

The generated clip IDs are timestamp-based and unique. The three rejected rows
are present in `input/rejected_clips.jsonl`. The root index manifest records the
source SHA-256, generated file hashes, spaCy model, schema/scorer versions, and
candidate field policies.

The final extraction run produced 8,899 action records from 1,022 distinct
action lemmas. Actions were found in 1,296 of 1,663 canonical clips (77.9%).
Objects were found in 64.0% of action records and directly attached tools in
7.4%. VerbNet covered 70.9% and FrameNet covered 69.3% of distinct actions.

The 8,899 count was checked after removing duplicate parsing of provenance
metadata. The final source-field counts are only:

```text
title:       1,255
description: 6,862
summary:       782
```

## Search checks

`remove old faucet` returned `Remove Faucet Stem` and `Remove Faucet Handle` as
the first two results for lexical, structured, and hybrid search.

`paint wall` exercised the terse-imperative parser fallback and returned
`Prepare and Paint Closet Walls` first.

`tighten screw with screwdriver` returned `Tighten the Set Screws` first, with
action, object, and tool signals all equal to 1.0.

The nominal query `faucet removal` did not produce a false structured action.
Hybrid search fell back to lexical results and included an explicit warning.

These are functional smoke tests, not accuracy estimates.

## Benchmark checks

The benchmark kept one fixed pool of 582 candidates and excluded clip titles
and parent-video text from every candidate representation. Forty-six terse
queries were recovered by the imperative fallback. After removing ambiguous
query names, exact phrase leaks, and remaining unparseable queries, all methods
were evaluated on the same 463 queries.

Whole-corpus results:

```text
method                 Hit@1   Hit@3   Hit@10   MRR
lexical TF-IDF          .631    .855     .933   .749
structured action       .261    .354     .475   .334
50/50 hybrid            .505    .616     .680   .577
```

For hybrid minus lexical Hit@1, the video-cluster bootstrap estimate was
-0.125 with a 95% interval of approximately [-0.184, -0.069]. The hybrid is
therefore worse on this whole-corpus proxy task.

In the within-video task, lexical and hybrid both reached Hit@1 = .726. The
hybrid-minus-lexical estimate was 0.000 with a 95% interval of approximately
[-0.051, 0.053]. This run detected no difference, but the interval still allows
modest harm or benefit and is not an equivalence test.

## Remaining manual check

`quality/manual_review_sample.csv` contains 60 extraction examples. Its
action/object/tool precision fields are intentionally blank. No parser
precision claim should be made until a person labels that worksheet.
