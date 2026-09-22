# Data provenance

Updated September 22, 2026, following the data owner's confirmation that the
supplied source data is authentic human-generated data. Descriptions of these
source annotations as machine-labeled or synthetic are incorrect.

## Source annotations

- Source: `indexed-videos-250.jsonl` in the repository root (private local file).
- Content: 250 videos and 1,703 nested clip annotations, including names,
  descriptions, goals, tools, supplies, and timestamps; some fields are empty.
- Provenance: authentic human-generated annotations, confirmed by the data owner.
- SHA-256: `96e93a9b51ab9f928615480a1ad89c90f19e00a43dd9153f20e003061f96b81c`.
- Cleaned derivative:
  `project1_outputs/indexed-video-sample/input/indexed_video_clips.jsonl`,
  containing 1,663 canonical clips after interval validation and deduplication.

This provenance statement applies to this supplied dataset, not automatically
to other exports passed to the CLI.

## Derived labels and evaluation scope

The pipeline automatically extracts action/object/tool records from the
human-generated annotations. Those extracted records still require a separate
precision review; the source's human authorship does not validate the parser.

The retrieval benchmark uses human-generated annotation fields to construct
queries and matching-clip targets. It is a field-held-out retrieval experiment
on authentic human-generated data, with possible dependence between fields.
It does not require a new A/B preference export to run.

The metadata-contrast and paraphrased hard-negative generators separately
construct requests, alternatives, and assigned winners. Their
`human_judgments: false` flags describe those generated A/B decisions, not the
authorship of the source annotations. Keep those flags intact.

Likewise, `human_relevance_labels_present: false` in the index profile refers
to separate comparative relevance judgments; it does not mean that the source
annotations are machine-generated.

No separate authentic W25 A/B preference export is currently present locally.
That export is needed for the human-preference agreement experiment, not for
the existing retrieval benchmark. Correcting source provenance does not change
the recorded scores or convert generated winners into human preference labels.
