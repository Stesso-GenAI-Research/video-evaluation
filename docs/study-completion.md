# Completing the study

Prepared October 15, 2026. This protocol separates development diagnosis from
independent evaluation. The paper draft is in `paper/main.tex` and its compiled
version is `paper/report.pdf`.

## Ready for human review

The packet at `project1_outputs/review-kit-2026-10-15/` contains:

- `extraction_review.csv`: 60 existing parser-review rows with source sentences,
  proposed action/object/tool terms, video links, timestamps, and blank labels.
- `preference_review.csv`: 44 diagnostic A/B cases with video links, intervals,
  query text, and blank preference/confidence/notes columns. Prior model scores
  and constructed winners are omitted; display order is randomized.
- `README.md`: reviewer instructions and limitations.
- `private/review_key.json`: protected row identities and selection metadata.
  Do not give this file to preference reviewers before judgments are complete.
- `diagnostic_evidence.json`: score-based evidence breakdown, not human labels.

Each selected preference case comes from a different source step. There are
eight query-parse-failure cases and twelve cases from each of zero structured
evidence, structured-only correctness under the constructed label, and hybrid
regression. Sampling was deterministic (seed 42). The groups are used for
selection, not presented as verified error causes.

The existing 60-row extraction sample was also selected by diagnostic groups;
some groups can overlap. Its precision should be reported by review group with
coverage, not represented as an unbiased corpus-wide estimate.

Prepare a new packet without overwriting an existing one:

```bash
.venv/bin/action-semantics prepare-review \
  --suite project1_outputs/experiment-suite-2026-10-06 \
  --output-dir project1_outputs/new-review-kit
```

## Score completed parser labels

Fill `action_correct`, `object_correct`, and `tool_correct` with yes/no.
Leave a value blank when it cannot be judged; document the reason in notes.
Review the source sentence rather than accepting the parser's confidence as
correctness. A direct tool is a grammatical attachment, distinct from a tool
listed somewhere in the clip inventory.

```bash
.venv/bin/action-semantics review \
  --review-csv project1_outputs/review-kit-2026-10-15/extraction_review.csv \
  --output-json project1_outputs/review-kit-2026-10-15/parser-results.json
```

## Export observed preference judgments

Inspect both video intervals for the given query. Enter A, B, tie, or unjudgeable;
blank means pending. Optional confidence ranges from 0 to 1. Use one worksheet
per reviewer and retain independent responses before any adjudication. A
text-only comparison should not silently substitute for a video judgment when
the clip cannot be viewed.

After filling the sheet:

```bash
.venv/bin/action-semantics export-reviewed-pairs \
  --kit project1_outputs/review-kit-2026-10-15 \
  --worksheet project1_outputs/review-kit-2026-10-15/preference_review.csv \
  --output-dir project1_outputs/review-export \
  --judge reviewer-01

./scripts/run_local_pipeline.sh evaluate-pairs \
  --index project1_outputs/experiment-suite-2026-10-06/index \
  --steps project1_outputs/review-export/steps.jsonl \
  --pairs project1_outputs/review-export/pairwise.jsonl \
  --output project1_outputs/review-export/evaluation \
  --seed 42 --bootstrap-iterations 5000
```

The exporter rejects edited query/clip columns, swapped clip identities,
duplicate/missing rows, and invalid confidence. It never infers a winner from
the original construction rule. It writes only explicit A/B decisions to
`pairwise.jsonl`; ties, unjudgeable cases, and pending rows are retained in
`excluded_judgments.jsonl` and counted in `export_summary.json`. An all-blank or
nonbinary worksheet does not create evaluator inputs.

These human judgments support diagnosis of selected development cases. They
do not become the primary W25 result or an independent held-out accuracy
estimate simply because a person supplied the labels.

## Prospective evaluation protocol

1. Treat all 250 existing videos as development-exposed. Retain their source
   hash and identifiers. Do not retrospectively call a subset untouched.
2. Before examining new outcomes, specify the new video/query population,
   sampling method, planned sample size, primary metric, acceptable regression,
   exclusions, confidence procedure, and stopping rule. The actual sample size
   and effect-size target still require a study decision; none is invented here.
3. Obtain new videos and independently written queries/preferences. Compare
   both stable video identifiers and content identities with development data.
   Related steps from one video must remain in one evaluation partition.
4. Check identifier overlap before scoring:

   ```bash
   .venv/bin/action-semantics check-holdout \
     --development-source indexed-videos-250.jsonl \
     --candidate-source data/prospective-videos.jsonl \
     --output-dir project1_outputs/prospective-overlap-check
   ```

   The command returns exit code 2 when overlaps or within-candidate duplicates
   occur, and writes a row-level report. New database IDs cannot hide a matching
   YouTube ID or normalized YouTube URL. Passing is only an identity check:
   reuploads, near-duplicate content, prior exposure, and query leakage still
   require review. Both input exports must contain video records.
5. Freeze parser code/model, lexicon, candidate fields, weights, and query rules.
   Record the code, dataset and configuration hashes. No tuning on new labels.
6. Validate references, then run the frozen evaluator once on observed A/B
   judgments. Report all delivered rows, unresolved references, ties/abstentions,
   analyzed counts, and step-clustered confidence intervals. If collecting
   top-k relevance labels instead, use the separate blinded ranking workflow.
7. Report the final result even if structured matching fails to outperform
   lexical matching. Any subsequent optimization requires a new holdout.

The guard was exercised on the existing export against itself: all 250 videos
overlap and it correctly refuses to treat them as prospective data. No final
held-out experiment has been performed.

## What the additional diagnostics establish

Of 415 comparisons with both structured scores equal to zero:

| Observed score condition | Comparisons |
|---|---:|
| Query has no parsed action | 21 |
| One candidate has no parsed actions | 58 |
| Query and both candidates have parses, but both selected action matches are zero | 336 |

The last group accounts for about 81% of these cases. The review should examine
the selected actions and their compatibility as well as parser failures.
Zero action match can also reflect negation or truly different actions; it is
not automatically a synonym error. The source per-corpus breakdown is preserved
in `diagnostic_evidence.json`. No scoring parameters were changed from these
observations.

## Remaining human decisions

- Author/affiliation and the description of the original annotation procedure.
- Actual parser and preference judgments, including any adjudication.
- Access to an independent evaluation population, sampling targets and approval
  of the prospective protocol.
- Data-release permissions and the final interpretation for submission.

The tooling, blank review packet, export validation, overlap guard, diagnostic
breakdown, and reproducible manuscript build are implemented. The unfinished
empirical sections remain explicit placeholders in the paper.
