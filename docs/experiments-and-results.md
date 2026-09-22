# Experiment methodology and interpretation

Current counts, numeric results, reproduction commands, output paths, and the
Stesso status summary are maintained in the
[project status, experiments, and execution guide](stesso-project-update.md).
This page defines the experiments and the limits on their interpretation.

## Evaluation types

The repository supports three distinct evaluations:

| Evaluation | Input labels | Question answered |
|---|---|---|
| Field-held-out retrieval benchmark | Fields paired within the indexed-video export | Can a method recover a clip from other annotation fields? |
| Pairwise preference evaluation | Already-observed A/B judgments | Does a frozen method agree with the judged winner? |
| Old-versus-new ranking review | Independent blinded relevance labels | Is a challenger top-*k* list better than a supplied historical list? |

Results from one evaluation must not be presented as results from another.

## Field-held-out retrieval benchmark

The supplied indexed-video annotations are authentic human-generated data,
confirmed by the data owner on September 22, 2026. See
[Data provenance](data-provenance.md). Automatic benchmark construction does
not imply machine-generated source annotations.

The automatic development benchmark uses:

```text
query        = clip name
known target = the same canonical clip
candidate    = description + goal + tools + supplies
```

The candidate representation excludes clip names and parent-video text. A clip
is excluded as a query when its normalized name is ambiguous, the exact
normalized phrase appears in its candidate fields, or no usable query action is
available. Excluded query clips remain in the candidate pool as distractors.

The benchmark reports Hit@1, Hit@3, Hit@10, and mean reciprocal rank. A target
receives a rank only when its score is positive. Exact score ties receive the
expected credit across their possible ranks rather than being resolved by clip
ID order. Uncertainty is estimated by resampling source-video clusters.

This is a controlled development task, not a human relevance evaluation. Clip
names and descriptions may retain partial wording or semantic dependence even
after direct phrase controls.

## Frozen pairwise-preference evaluation

The primary Project 1, Step 1 experiment accepts an existing project step, two
specified clips, and an observed A/B winner. It computes direct scores:

```text
score(step query, clip A)
score(step query, clip B)
```

It does not retrieve challengers or infer preference from whole-corpus rank.
Agreement credit is `1.0` for a higher winner score, `0.5` for an exact tie, and
`0.0` for a higher loser score. Confidence intervals resample complete
`step_id` clusters because comparisons for one project step are related.

The primary result includes all valid delivered judgments. Judge provenance,
confidence, and winner position are retained. Provenance slices are descriptive
secondary results and are not used for tuning.

See [Pairwise preference evaluation](pairwise-preference-evaluation.md) for the
fixed query rule, input contracts, resolution behavior, validation, and output
schema.

## Constructed pairwise development data

The repository can generate:

- a metadata-contrast corpus selected from parsed action, object, context, and
  source-video metadata without consulting retrieval scores; and
- a paraphrased hard-negative corpus that deliberately uses frozen scorer
  outputs to select difficult alternatives.

In both cases, the target clip supplies the assigned winner. These datasets are
appropriate for regression tests, pipeline validation, scorer stress tests, and
case sampling. They do not estimate human or production-cascade agreement. The
two aggregate rates must not be pooled because their selection mechanisms
differ.

Every generated artifact records that it is constructed development data and
not research evidence. Constructed rows must never be relabeled as authentic
human judgments.

## Old-versus-new ranking review

This experiment is separate from pairwise preference evaluation. It requires a
supplied historical ranking for each query or step, including stable IDs,
original ranks, and canonical clip IDs or resolvable timestamps.

The workflow is:

1. Validate and resolve all supplied historical references.
2. Generate one challenger top-*k* list per query.
3. Pool and shuffle both lists under a hidden A/B assignment.
4. Record independent overall, action, object, and tool relevance labels.
5. Compare Precision@*k*, Success@*k*, wins/ties/losses, and a paired confidence
   interval.

Ranking overlap is descriptive and never determines a winner. The worksheet
must be judged independently. See
[Running the pipeline](running-the-pipeline.md#comparing-rankings) for commands.

## Interpretation rules

- Preference agreement is not top-*k* corpus retrieval accuracy.
- Constructed-label recovery is not human preference agreement.
- Parser field coverage is not parser precision.
- Equal observed metrics do not establish statistical equivalence.
- A few plausible search results are functional smoke checks, not accuracy
  evidence.
- Missing or ambiguous clip references remain validation outcomes; they are not
  silently removed from reported totals.
- The frozen W25 evaluation must run without selecting a query form, parser
  behavior, threshold, structured weight, or hybrid alpha from observed W25
  results.

## Research sequence

1. Run the frozen Step 1 evaluation on the authentic W25 export.
2. Preserve the complete pair-level output and primary all-judgment result.
3. Perform Step 2 descriptive error analysis only after Step 1 is recorded.
4. Develop parser, tie, or hybrid changes on declared development data.
5. Evaluate any later optimized system once on a separate held-out set.

The current numeric conclusion and external data dependency are stated in the
[canonical project guide](stesso-project-update.md).
