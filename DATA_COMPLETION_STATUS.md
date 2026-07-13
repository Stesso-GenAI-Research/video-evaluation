# Data completion status

This repository now has a working structured search path for `indexed-videos-250.jsonl`, a private sample of 250 indexed videos with 1,703 nested clip records. Each nested clip is an instructional step with a name, metadata, and timestamps, so the file is enough to build an index and return top-k action-semantic matches.

Dense embeddings and pairwise judgments are still useful for later comparative evaluation, but they are not required for the immediate structured-search deliverable.

What is complete in this repository:

- The month 1 extraction code is present.
- The month 2 FrameNet, SRL-style extraction, and taxonomy code is present.
- The month 3 scoring and pairwise evaluation code is present.
- The data contracts and SQL export guide are included.
- The repository can be installed and tested locally.
- The new `prepare-indexed-videos` command flattens the nested sample into stable clip records and writes a data-availability profile.
- The new `run-indexed-video-analysis` command builds and verifies the structured index from the real sample.
- The real sample run completed locally. It produced 9,246 action triples, 1,023 distinct action lemmas, and a 670-action, 80-cluster first-pass taxonomy.
- The pipeline now measures extraction quality automatically. Actions were found in 1,335 of 1,703 clips (78.4%), VerbNet covered 70.9% of action lemmas, and FrameNet covered 69.2%.
- A deterministic 60-row manual-review worksheet is generated at `quality/manual_review_sample.csv`.
- Record-level tool metadata is preserved separately from directly parsed tools and can be used as a fallback in structured scoring.
- The `search-indexed-clips` command parses an action query, scores all 1,703 clips, and returns the top matches.
- The Bash script exposes this as `./scripts/run_local_pipeline.sh search "query text"` and saves the result as JSON.
- The `compare-indexed-clips` command compares supplied original matches, or a TF-IDF baseline, with the new action-semantic top three.
- The comparison reports overlap plus text, action, object, tool, VerbNet, FrameNet, taxonomy, and total structured scores for both sets.

What should happen next:

- Run and inspect realistic top-three searches.
- Review a stratified sample of the extraction output manually before making accuracy claims.
- Label the generated review worksheet, summarize the error types, and report human-reviewed action/object/tool precision.
- Build an aligned-clip benchmark from the nested records.
- Request a larger export in the same IndexedVideo format.
- Add dense or lexical baselines and compare them with structured and hybrid scoring.

The current file is sufficient for retrieval and qualitative top-k inspection. Human labels would strengthen a later effectiveness study, but their absence no longer blocks the working search system.
