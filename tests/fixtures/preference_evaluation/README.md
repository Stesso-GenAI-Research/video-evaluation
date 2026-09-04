# Synthetic pairwise-preference fixtures

These tiny files exist only for automated tests and local smoke checks. They
are fabricated examples, not W25 source data, and their outputs are not
research results.

- `indexed-videos-synthetic.jsonl` contains three parent videos and six nested
  segments in the strict `IndexedVideo` source format. Canonical IDs are
  derived as `indexed-video-{video_id}-segment-{start}-{end}`.
- `steps-synthetic.jsonl` contains three synthetic project steps.
- `pairs-synthetic.jsonl` contains four judgments, including A and B winners,
  timestamp and exact-ID references, cascade and human provenance, confidence
  values, two comparisons sharing one step, and multiple steps.

The two tile-cleaning segments intentionally have identical searchable fields
at different timestamps. `synthetic-comparison-004` therefore exercises exact
method ties while still resolving A and B to distinct canonical clip IDs.
