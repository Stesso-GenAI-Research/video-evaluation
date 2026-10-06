"""Reproducible local development experiments with separate, paired comparisons."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .config import DEFAULT_SPACY_MODEL
from .contrast_preferences import generate_contrast_preferences
from .io_utils import sha256_file
from .logging_utils import info
from .pair_diagnostics import analyze_pair_scores
from .provenance import build_manifest, write_manifest
from .retrieval.benchmark import (
    METHODS, PRIMARY_METRICS, RankOutcome, _metrics,
    _paired_cluster_bootstrap_delta_cis, run_field_heldout_benchmark,
)
from .retrieval.preference_evaluation import run_preference_evaluation
from .sample_analysis import run_indexed_video_analysis
from .synthetic_preferences import generate_controlled_preferences

ARMS = {
    "baseline": (False, False), "aliases": (True, False),
    "primary": (False, True), "combined": (True, True),
}
CORPORA = ("metadata-contrast-preference-development", "paraphrased-preference-development")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_benchmark(directory: Path) -> tuple[dict, list[dict], dict]:
    summary = _read(directory / "benchmark_summary.json")
    with (directory / "benchmark_queries.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ids = [r["query_clip_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids) or len(rows) != summary["query_count"]:
        raise ValueError(f"Empty, duplicate or inconsistent query cohort: {directory}")
    ranks = {}
    for method in METHODS:
        outcomes = []
        for row in rows:
            def integer(suffix):
                value = row[f"{method}_{suffix}"]
                return int(value) if value else None
            lo, hi = integer("best_tied_rank"), integer("worst_tied_rank")
            tie_size = integer("tie_size")
            if (lo is None) != (hi is None) or (
                lo is not None and (lo < 1 or hi < lo or tie_size != hi - lo + 1)
            ):
                raise ValueError(f"Invalid tied rank interval in {directory}")
            outcomes.append(RankOutcome(
                deterministic_rank=integer("deterministic_rank"), best_rank=lo,
                worst_rank=hi, tie_size=tie_size or 0,
                relevant_score=float(row[f"{method}_relevant_score"]),
                positive_candidate_count=integer("positive_candidate_count") or 0,
                top_clip_id=row[f"{method}_top_clip_id"] or None,
            ))
        calculated = _metrics(outcomes)
        if any(not math.isclose(calculated[m], summary["methods"][method][m], abs_tol=1e-12)
               for m in PRIMARY_METRICS):
            raise ValueError(f"CSV ranks disagree with summary metrics: {directory}/{method}")
        ranks[method] = outcomes
    return summary, rows, ranks


def compare_benchmark_arms(
    directories: dict[str, Path], *, iterations: int = 2000, seed: int = 1729,
) -> dict:
    """Reject mismatched cohorts/artifacts; bootstrap paired, tie-aware deltas."""
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    baseline, rows, ranks = _read_benchmark(directories["baseline"])

    def cohort(values):
        return [(r["query_clip_id"], r["query_text"], r["relevant_clip_id"], r["video_id"],
                 r["global_candidate_count"]) for r in values]

    result = {"query_count": len(rows), "candidate_count": baseline["corpus_count"],
              "same_ordered_queries": True, "arms": {}}
    for name, directory in directories.items():
        summary, other_rows, other_ranks = _read_benchmark(directory)
        if cohort(rows) != cohort(other_rows) or (
            baseline["reproducibility"] != summary["reproducibility"]
            or baseline["hybrid_alpha_lexical"] != summary["hybrid_alpha_lexical"]
        ):
            raise ValueError(f"Cannot compare different query cohorts, indexes or settings: {name}")
        result["arms"][name] = {
            "terminology": summary["terminology"], "methods": summary["methods"],
            "normalized_exact_phrase_query_ids": summary["normalized_exact_phrase_query_ids"],
            "paired_deltas_vs_baseline": {
                method: _paired_cluster_bootstrap_delta_cis(
                    video_ids=[row["video_id"] or row["query_clip_id"] for row in rows],
                    baseline_ranks=ranks[method], challenger_ranks=other_ranks[method],
                    iterations=iterations, seed=seed,
                ) for method in METHODS
            } if name != "baseline" else {},
        }
    return result


def _report(summary: dict) -> str:
    lines = ["# Local experiment suite", "", "Status: complete (development experiments).",
             "", "Human-written source annotations; pairwise labels below are constructed.",
             "No claim of human preference agreement or production improvement is made.", "",
             f"Retrieval: {summary['retrieval']['query_count']} identical queries against "
             f"{summary['retrieval']['candidate_count']} candidates.", "",
             "| Arm | Lexical Hit@1 | Structured Hit@1 | Hybrid Hit@1 |",
             "|---|---:|---:|---:|"]
    for arm, values in summary["retrieval"]["arms"].items():
        metrics = values["methods"]
        rates = " | ".join(f"{metrics[m]['hit_at_1']:.2%}" for m in METHODS)
        lines.append(f"| {arm} | {rates} |")
    lines += ["", "Pairwise agreement (ties receive half credit):", "",
              "| Corpus | Resolved / total | Lexical | Structured | Hybrid |",
              "|---|---:|---:|---:|---:|"]
    for name, values in summary["pairwise"].items():
        c = values["counts"]
        rates = " | ".join(f"{values['methods'][m]['mean_agreement_credit']:.2%}"
                           for m in ("lexical", "structured", "hybrid"))
        lines.append(f"| {name} | {c['resolved_comparisons']} / {c['total_comparisons']} | {rates} |")
    lines += ["", "Full confidence intervals, configurations and diagnostics: summary.json.",
              "Per-query and per-pair files remain in their respective directories.", "",
              "Still needed for research completion:", "",
              "- Independent human A/B judgments and step text for the primary preference test.",
              "- Completed extraction-quality review to measure parser precision.",
              "- A held-out evaluation after any further development/tuning.", ""]
    return "\n".join(lines)


def run_experiment_suite(
    *, source: Path, output_dir: Path, development_root: Path | None = None,
    spacy_model: str = DEFAULT_SPACY_MODEL, seed: int = 42,
    bootstrap_iterations: int = 5000,
) -> dict[str, Path]:
    """Build once, run four retrieval arms and two fixed-label development tests.

    An existing output directory is never reused. Supplied development inputs
    are copied verbatim; without them, generate new, explicitly constructed
    corpora. Never silently replace a missing supplied file with generated data.
    """
    if bootstrap_iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    if not source.is_file():
        raise FileNotFoundError(source)
    source_inputs = [source]
    if development_root is not None:
        for corpus in CORPORA:
            for filename in ("steps.jsonl", "pairwise.jsonl", "generation_manifest.json"):
                path = development_root / corpus / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                source_inputs.append(path)
            manifest = _read(development_root / corpus / "generation_manifest.json")
            if manifest.get("parameters", {}).get("human_judgments") is not False:
                raise ValueError(f"Expected explicitly constructed development corpus: {corpus}")
    output_dir.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "completed_stages": [], "current_stage": "build-index"}

    def stage(name, operation):
        status["current_stage"] = name
        _write(output_dir / "status.json", status)
        info(f"Experiment suite: {name}")
        value = operation()
        status["completed_stages"].append(name)
        _write(output_dir / "status.json", status)
        return value

    try:
        index = output_dir / "index"
        stage("build-index", lambda: run_indexed_video_analysis(
            indexed_videos_jsonl=source, output_dir=index, spacy_model=spacy_model,
            random_seed=1729, min_taxonomy_support=2,
        ))
        directories = {}
        for name, (aliases, primary) in ARMS.items():
            directories[name] = output_dir / "retrieval" / name
            stage(f"retrieval-{name}", lambda: run_field_heldout_benchmark(
                clips_jsonl=index / "input/indexed_video_clips.jsonl",
                month1_dir=index / "month1", month2_dir=index / "month2",
                output_dir=directories[name], spacy_model=spacy_model,
                terminology=aliases, primary_inventory_only=primary,
            ))
        comparison = stage("paired-retrieval-comparison", lambda: compare_benchmark_arms(directories))
        pairwise = {}
        for corpus in CORPORA:
            destination = output_dir / "pairwise" / corpus
            if development_root is None:
                generator = (generate_contrast_preferences if corpus == CORPORA[0]
                             else generate_controlled_preferences)
                stage(f"generate-{corpus}", lambda: generator(
                    index=index, output_dir=destination,
                    step_count=800 if corpus == CORPORA[0] else 350, seed=seed,
                ))
            else:
                destination.mkdir(parents=True)
                for filename in ("steps.jsonl", "pairwise.jsonl", "generation_manifest.json"):
                    shutil.copyfile(development_root / corpus / filename, destination / filename)
            stage(f"evaluate-{corpus}", lambda: run_preference_evaluation(
                index=index, steps_jsonl=destination / "steps.jsonl",
                pairs_jsonl=destination / "pairwise.jsonl", output_dir=destination / "step1",
                seed=seed, bootstrap_iterations=bootstrap_iterations,
            ))
            stage(f"diagnose-{corpus}", lambda: analyze_pair_scores(
                pair_scores_jsonl=destination / "step1/pair_scores.jsonl",
                output_dir=destination / "diagnostics",
            ))
            pairwise[corpus] = _read(destination / "step1/summary.json")
            pairwise[corpus]["diagnostics"] = _read(
                destination / "diagnostics/diagnostic_summary.json")
        summary = {
            "schema_version": "experiment-suite.v1", "status": "complete",
            "source_sha256": sha256_file(source), "retrieval": comparison,
            "pairwise": pairwise,
            "development_inputs": "copied_verbatim" if development_root else "newly_constructed",
            "data_profile": _read(index / "input/indexed_video_profile.json"),
            "extraction_quality": _read(index / "quality/extraction_quality_summary.json"),
            "human_preference_evaluation": "not_run; requires separate observed judgments",
        }
        _write(output_dir / "summary.json", summary)
        (output_dir / "summary.md").write_text(_report(summary), encoding="utf-8")
        code_paths = sorted(Path(__file__).parent.rglob("*.py"))
        manifest = build_manifest(
            command="experiment-suite", input_files=source_inputs + code_paths,
            output_files=sorted(p for p in output_dir.rglob("*")
                                if p.is_file() and p.name != "status.json"),
            parameters={"seed": seed, "pairwise_bootstrap_iterations": bootstrap_iterations,
                        "spacy_model": spacy_model, "development_inputs": summary["development_inputs"],
                        "retrieval_bootstrap_seed": 1729, "retrieval_bootstrap_iterations": 2000},
        )
        write_manifest(output_dir / "manifest.json", manifest)
        status.update(status="complete", current_stage=None)
        _write(output_dir / "status.json", status)
        return {"summary": output_dir / "summary.json", "report": output_dir / "summary.md",
                "manifest": output_dir / "manifest.json"}
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write(output_dir / "status.json", status)
        raise
