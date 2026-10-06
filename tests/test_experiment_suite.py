import csv
import json

import pytest
from typer.testing import CliRunner

from action_semantics.cli import app
from action_semantics.experiment_suite import compare_benchmark_arms, run_experiment_suite
from action_semantics.retrieval.benchmark import METHODS, RankOutcome, _metrics


def _benchmark(directory, *, video="video", duplicate=False, best=1, worst=2):
    directory.mkdir()
    outcome = RankOutcome(best, best, worst, "clip", 0.5, 3, worst - best + 1)
    summary = {
        "query_count": 2 if duplicate else 1, "corpus_count": 3,
        "hybrid_alpha_lexical": 0.5, "reproducibility": {"clips_sha256": "same"},
        "terminology": {}, "normalized_exact_phrase_query_ids": [],
        "methods": {m: _metrics([outcome] * (2 if duplicate else 1)) for m in METHODS},
    }
    row = {"query_clip_id": "clip", "relevant_clip_id": "clip", "query_text": "cut",
           "video_id": video, "global_candidate_count": 3}
    for method in METHODS:
        for key, value in {
            "deterministic_rank": best, "best_tied_rank": best, "worst_tied_rank": worst,
            "tie_size": worst - best + 1, "relevant_score": 0.5,
            "positive_candidate_count": 3, "top_clip_id": "clip",
        }.items():
            row[f"{method}_{key}"] = value
    (directory / "benchmark_summary.json").write_text(json.dumps(summary))
    with (directory / "benchmark_queries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows([row] * (2 if duplicate else 1))
    return directory


def test_comparison_uses_fractional_tie_credit(tmp_path):
    baseline = _benchmark(tmp_path / "baseline")
    challenger = _benchmark(tmp_path / "challenger", best=1, worst=1)
    report = compare_benchmark_arms({"baseline": baseline, "aliases": challenger}, iterations=10)
    delta = report["arms"]["aliases"]["paired_deltas_vs_baseline"]["lexical_tfidf"]
    assert delta["metrics"]["hit_at_1"]["estimate"] == 0.5
    assert delta["metrics"]["mean_reciprocal_rank"]["estimate"] == 0.25
    assert delta["cluster_unit"] == "video"


def test_comparison_rejects_changed_video_cluster(tmp_path):
    baseline = _benchmark(tmp_path / "baseline")
    challenger = _benchmark(tmp_path / "challenger", video="different")
    with pytest.raises(ValueError, match="different query cohorts"):
        compare_benchmark_arms({"baseline": baseline, "aliases": challenger}, iterations=10)


def test_comparison_rejects_duplicate_queries(tmp_path):
    baseline = _benchmark(tmp_path / "baseline", duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        compare_benchmark_arms({"baseline": baseline}, iterations=10)


def test_comparison_rejects_stale_summary_metrics(tmp_path):
    baseline = _benchmark(tmp_path / "baseline")
    path = baseline / "benchmark_summary.json"
    summary = json.loads(path.read_text())
    summary["methods"]["lexical_tfidf"]["hit_at_1"] = 1.0
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="CSV ranks disagree"):
        compare_benchmark_arms({"baseline": baseline}, iterations=10)


def test_suite_never_overwrites_results(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "summary.md"
    sentinel.write_text("Reviewed results")
    with pytest.raises(FileExistsError):
        run_experiment_suite(source=source, output_dir=output)
    assert sentinel.read_text() == "Reviewed results"


def test_supplied_missing_corpus_is_not_silently_regenerated(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    output = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        run_experiment_suite(source=source, output_dir=output, development_root=tmp_path)
    assert not output.exists()


def test_failure_records_stage_and_cannot_be_mistaken_for_completion(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    def fail(**kwargs):
        raise RuntimeError("missing model")
    monkeypatch.setattr("action_semantics.experiment_suite.run_indexed_video_analysis", fail)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="missing model"):
        run_experiment_suite(source=source, output_dir=output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["current_stage"] == "build-index"
    assert status["completed_stages"] == []
    assert not (output / "summary.md").exists()


def test_cli_requires_new_output_and_exposes_development_inputs():
    result = CliRunner().invoke(app, ["experiment-suite", "--help"])
    assert result.exit_code == 0
    assert "development" in result.stdout
