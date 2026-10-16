"""Generate manuscript tables from a completed, hash-verified experiment run."""

import argparse
import hashlib
import json
from pathlib import Path


def generate(suite: Path, output: Path, review_results: Path | None = None) -> None:
    review_tex = (r"\newcommand{\HumanReviewResults}{\pending{Diagnostic extraction and preference "
                  r"judgments have not yet been returned.}}" + "\n")
    if review_results is not None:
        review_manifest = json.loads((review_results / "manifest.json").read_text())
        for name in ("analysis.json", "review-results.tex"):
            digest = hashlib.sha256((review_results / name).read_bytes()).hexdigest()
            if digest != review_manifest["outputs"].get(name):
                raise ValueError(f"Review result hash changed: {name}")
        review = json.loads((review_results / "analysis.json").read_text())
        if review.get("status") != "complete" or review.get("purpose") != "diagnostic_human_review_pilot":
            raise ValueError("Only a completed diagnostic review can populate the manuscript")
        review_tex = (review_results / "review-results.tex").read_text()
    manifest = json.loads((suite / "manifest.json").read_text())
    summary_path = suite / "summary.json"
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    if not any(row["sha256"] == digest and Path(row["path"]).name == "summary.json"
               for row in manifest["outputs"]):
        raise ValueError("Suite summary does not match its recorded manifest")
    summary = json.loads(summary_path.read_text())
    if review_results is not None and review.get("source_suite_summary_sha256") != digest:
        raise ValueError("Review packet and manuscript use different experiment suites")
    if summary["status"] != "complete":
        raise ValueError("Only completed experiment runs can populate the manuscript")
    baseline_path = suite / "retrieval/baseline/benchmark_summary.json"
    baseline_digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    if not any(row["sha256"] == baseline_digest for row in manifest["outputs"]):
        raise ValueError("Baseline report does not match its recorded manifest")
    baseline = json.loads(baseline_path.read_text())
    lines = [f"% Source suite: {suite.name}", f"% Summary SHA-256: {digest}"]

    def macro(name, value):
        lines.append("\\newcommand{\\" + name + "}{" + str(value) + "}")

    def table(name, rows):
        macro(name, "\n" + "\n".join(" & ".join(row) + r" \\" for row in rows) + "\n")

    def percent(x):
        return f"{100*x:.2f}"
    labels = {"lexical_tfidf": "Lexical TF--IDF", "structured_action": "Structured",
              "hybrid": "Hybrid (50/50)"}
    profile = summary["data_profile"]
    quality = summary["extraction_quality"]
    for key, val in {"VideoCount": profile["video_count"],
                     "ClipCount": f"{profile['canonical_clip_count']:,}",
                     "QueryCount": baseline["query_count"],
                     "CandidateCount": baseline["corpus_count"],
                     "LexicalHitOne": f"{100*baseline['methods']['lexical_tfidf']['hit_at_1']:.1f}",
                     "StructuredHitOne": f"{100*baseline['methods']['structured_action']['hit_at_1']:.1f}",
                     "HybridHitOne": f"{100*baseline['methods']['hybrid']['hit_at_1']:.1f}"}.items():
        macro(key, val)
    table("InventoryRows", [
        ["Parent videos", str(profile["video_count"])],
        ["Raw clip annotations", f"{profile['raw_clip_count']:,}"],
        ["Valid annotations", f"{profile['valid_source_clip_count']:,}"],
        ["Rejected intervals", str(profile["rejected_clip_count"])],
        ["Canonical clips", f"{profile['canonical_clip_count']:,}"],
        ["Extracted action records", f"{quality['triple_count']:,}"],
        ["Clips with at least one extracted action",
         f"{quality['clips_with_extracted_action']:,} ({100*quality['clip_action_coverage']:.1f}\\%)"],
        ["Action records with an object", f"{100*quality['triple_object_coverage']:.1f}\\%"],
        ["Action records with a directly attached tool",
         f"{100*quality['direct_tool_coverage']:.1f}\\%"],
    ])
    for task, macro_name in [("global", "GlobalRows"), ("within_video", "WithinRows")]:
        table(macro_name, [[labels[m]] + [
            percent(baseline["tasks"][task]["methods"][m][metric])
            for metric in ("hit_at_1", "hit_at_3", "hit_at_10")
        ] + [f"{baseline['tasks'][task]['methods'][m]['mean_reciprocal_rank']:.3f}"]
                          for m in labels])
    arms = [("baseline", "Cleaned baseline"), ("aliases", "Aliases only"),
            ("primary", "Primary inventory only"), ("combined", "Both changes")]
    table("AblationRows", [[label] + [percent(summary["retrieval"]["arms"][arm]["methods"][m]["hit_at_1"])
                                     for m in labels] for arm, label in arms])
    table("DeltaRows", [[label] + [f"{100*v:.2f}" for v in (
        summary["retrieval"]["arms"][arm]["paired_deltas_vs_baseline"]["lexical_tfidf"]["metrics"]["hit_at_1"][key]
        for key in ("estimate", "ci_95_low", "ci_95_high"))]
                       for arm, label in arms if arm != "baseline"])
    names = [("metadata-contrast-preference-development", "Metadata contrast"),
             ("paraphrased-preference-development", "Hard negatives")]
    pair_rows = []
    for name, label in names:
        result = summary["pairwise"][name]
        for m, method in [("lexical", "Lexical"), ("structured", "Structured"), ("hybrid", "Hybrid")]:
            row = result["methods"][m]
            ci = row["clustered_bootstrap_95_ci"]
            pair_rows.append([label, method, f"{row['resolved_pairs']:,}",
                              percent(row["mean_agreement_credit"]),
                              f"[{percent(ci['lower'])}, {percent(ci['upper'])}]"])
    table("PairwiseRows", pair_rows)
    flags = [("lexical_structured_disagreement", "Lexical/structured disagreement"),
             ("structured_tie", "Structured ties"), ("structured_zero_evidence", "Zero structured evidence"),
             ("query_parse_failure", "Query parse failure"),
             ("hybrid_recovery_vs_lexical", "Hybrid recovery relative to lexical"),
             ("hybrid_regression_vs_lexical", "Hybrid regression relative to lexical")]
    table("DiagnosticRows", [[label] + [str(summary["pairwise"][name]["diagnostics"]["diagnostic_flag_counts"][flag])
                                        for name, _ in names] for flag, label in flags])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output.parent / "review-results.tex").write_text(review_tex, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/generated/results.tex"))
    parser.add_argument("--review-results", type=Path)
    args = parser.parse_args()
    generate(args.suite, args.output, args.review_results)
