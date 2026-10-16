# Completing the study

Updated October 16, 2026. The remaining local workflow is ready: returned review
sheets can be validated, analyzed against the recorded model scores, and included
in the manuscript without writing another analysis script. The current packet
supports diagnosis of existing development cases. Completing it will not supply
the independent preference data needed for the broader research claim.

## Give reviewers the current packet

Use `project1_outputs/review-kit-ready-2026-10-15/`. Its `reviewer-packet.zip`
contains only the two blank worksheets and reviewer instructions. Give each
reviewer a separate copy of that ZIP, or of its three files. Keep
`private/`, the manifest, and diagnostic reports with the researcher until
judgments are complete. The worksheets contain video links and timestamps but
no model scores or constructed winners. A/B order is randomized.

There are 60 extraction examples and 44 preference cases, with one comparison
per selected source step. Eight preference cases concern query parse failures;
twelve come from each of three other diagnostic groups. Those groups describe
why cases were selected, not a verified cause of error. Extraction groups can
overlap, so their results must be reported by group rather than treated as a
random sample of the whole corpus.

For extraction, reviewers enter `yes`, `no`, or `unjudgeable` in each correctness
column. They should compare the proposed terms with the source sentence. A
missing tool is correct when the sentence does not state a direct tool. An
inventory item alone does not establish a grammatical attachment to an action.
For preferences, reviewers inspect both video intervals and enter `A`, `B`,
`tie`, or `unjudgeable`. Confidence is optional and ranges from 0 to 1. Blank
answers mean pending in both sheets. Notes should explain uncertainty. Reviewers
may reorder rows, but should preserve IDs, query text, and clip fields.

The version 2 packet adds extraction IDs, a hashed answer key, and the original
scores aligned with each randomized A/B pair. The original October 15 packet
remains untouched. Do not replace an already labeled sheet with a new blank one.
The old `review` command accepts yes/no and blanks; use `analyze-review` for the
version 2 sheets and their explicit `unjudgeable` responses.

To prepare another packet from a completed suite:

```bash
.venv/bin/action-semantics prepare-review \
  --suite project1_outputs/experiment-suite-2026-10-06 \
  --output-dir project1_outputs/new-review-kit
```

## Analyze returned sheets and rebuild the paper

Keep the returned files with their reviewer identity. For example, if the
reviewer returns files under `project1_outputs/reviewer-01/`, run:

```bash
.venv/bin/action-semantics analyze-review \
  --kit project1_outputs/review-kit-ready-2026-10-15 \
  --extraction project1_outputs/reviewer-01/extraction_review.csv \
  --preferences project1_outputs/reviewer-01/preference_review.csv \
  --judge reviewer-01 \
  --output-dir project1_outputs/completed-review

bash paper/build.sh \
  project1_outputs/experiment-suite-2026-10-06 \
  project1_outputs/completed-review
```

Use a new output directory. The first command rejects changed identities,
missing or duplicated rows, invalid labels, and pending answers. Add
`--allow-partial` only to inspect progress; partial results cannot populate the
paper. An explicitly unjudgeable answer is a completed response, not a correct
answer. A packet containing only ties or abstentions produces accounting and
undefined binary agreement, rather than invented preferences.

`analysis.json` and `analysis.md` contain extraction correctness and coverage by
group, preference response counts, method agreement, and paired differences
from lexical matching. Exact model-score ties receive half credit. Human ties
and abstentions are excluded from binary agreement and counted separately.
Confidence is retained but does not weight the result. The intervals use 5,000
bootstrap draws over source-step clusters with seed 42; they do not remove the
selection bias of this diagnostic sample. The JSON also reports descriptive
agreement by selection group.

`case_results.jsonl` allows inspection of each comparison. When explicit A/B
choices exist, `observed-pairs/` contains validated evaluator inputs and the
excluded-response inventory. The manifest records input, analysis-code, and
output hashes. The analysis uses scores saved before review, so no model is
refitted and no method setting changes after seeing the answers.

The build verifies the result hashes and checks that the packet came from the
same experiment suite as the manuscript. It inserts `review-results.tex` into the
paper's human-evaluation section. It does not fill the independent-test section
or rewrite the discussion automatically. Read the result and revise its
interpretation before submission. A normal `bash paper/build.sh` without the
second argument restores the diagnostic placeholder.

## Compare independent reviewers

If two people review the same preference sheet, preserve both responses before
discussing disagreements:

```bash
.venv/bin/action-semantics compare-reviewers \
  --kit project1_outputs/review-kit-ready-2026-10-15 \
  --first project1_outputs/reviewer-01/preference_review.csv \
  --second project1_outputs/reviewer-02/preference_review.csv \
  --output-dir project1_outputs/reviewer-agreement
```

`agreement.json` reports raw agreement and Cohen's kappa over jointly judged
A/B/tie responses. Pending and unjudgeable responses are excluded and counted.
Kappa remains undefined when there are no jointly judged cases or no variation
in the expected agreement. `disagreements.csv` includes both responses and blank
adjudication fields. The program does not select a label. After adjudication,
copy the agreed decisions into a separate complete preference worksheet with
the original columns, preserve the decision notes, and analyze it using a judge
identifier such as `adjudicated-panel`. Keep the independent analyses as well.
Extraction correctness can be analyzed separately for each reviewer; this
comparison command concerns preferences only.

## Finish independent evaluation

The existing W25 evaluator already accepts observed preferences and step text;
its [input contracts and command](pairwise-preference-evaluation.md) are ready.
If a new evaluation population is collected, document its sampling method,
sample size, primary metric, acceptable regression, exclusions, confidence
procedure, and stopping rule before examining outcomes. These study decisions
must be made with the people collecting the data. No sample size is assumed.

Treat all 250 current videos as development-exposed. Obtain new videos and
independently authored queries or preferences, then check their identities:

```bash
.venv/bin/action-semantics check-holdout \
  --development-source indexed-videos-250.jsonl \
  --candidate-source data/prospective-videos.jsonl \
  --output-dir project1_outputs/prospective-overlap-check
```

The check returns exit code 2 for development overlap or duplicate candidate
videos. It compares database IDs, YouTube IDs, and normalized URLs, so a new
database ID cannot hide the same YouTube video. Passing establishes only
disjoint identifiers. Reuploads, related content, prior exposure, and query
leakage still require review. Keep related steps from a video in one partition.

Freeze the parser, model, lexicon, candidate fields, weights, and query rules
before scoring the new judgments. Record their hashes and run the preference
evaluator once. Report delivered and analyzed counts, unresolved references,
ties, abstentions, and confidence intervals even if the result still favors
lexical matching. Later changes require another evaluation set.

## Current evidence and remaining decisions

The October 6 experiments still favor lexical retrieval. Of 415 comparisons
where both structured scores are zero, 21 have no query action, 58 lack actions
for one candidate, and 336 have parsed actions but zero selected action match.
The last group accounts for about 81% of these comparisons. It directs review
toward the extracted actions and their compatibility, without establishing
that synonyms caused the problem. No scoring parameters have changed.

The blank version 2 packet was analyzed as a readiness check. It reports 224
pending responses: 180 extraction judgments and 44 preferences. It reports no
human accuracy; the report is in `project1_outputs/review-readiness-2026-10-16/`.
The overlap guard also correctly rejects all 250 development
videos when the existing export is compared with itself.

The author and reviewers still need to supply actual judgments, the original
annotation procedure, author information, data-access conditions, and the
independent evaluation data and sampling decisions. Once those inputs arrive,
the local diagnostic analysis and paper-table handoff require the two commands
above. The final discussion still requires interpretation of the observed
results; the diagnostic packet alone cannot close the independent study.
