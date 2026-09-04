from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from action_semantics import cli
from action_semantics.io_utils import write_jsonl
from action_semantics.models import ActionTriple
from action_semantics.retrieval import preference_evaluation
from action_semantics.retrieval.preference_evaluation import (
    FROZEN_HYBRID_ALPHA,
    STEP_QUERY_RULE_VERSION,
    PreferencePairInput,
    PreferenceStepInput,
    build_step_query,
    exact_pair_outcome,
    run_preference_evaluation,
)
from action_semantics.retrieval.scorers import (
    StructuredResources,
    score_query_clip_ids,
)


def _triple(
    record_type: str,
    record_id: str,
    action: str,
    obj: str,
    *,
    tool: str | None = None,
) -> ActionTriple:
    return ActionTriple(
        record_type=record_type,
        record_id=record_id,
        source_field="synthetic_test_fixture",
        action=action,
        action_lemma=action,
        action_text=action,
        object_text=obj,
        object_lemmas=[obj],
        tool_text=tool,
        tool_lemmas=[tool] if tool else [],
        sentence=f"{action} the {obj}",
        extraction_method="synthetic_test_fixture",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_built_index(tmp_path: Path) -> Path:
    """Write the smallest valid built index used by evaluator tests."""
    root = tmp_path / "synthetic-built-index"
    input_dir = root / "input"
    month1_dir = root / "month1"
    month2_dir = root / "month2"
    input_dir.mkdir(parents=True)
    month1_dir.mkdir()
    month2_dir.mkdir()

    write_jsonl(
        input_dir / "indexed_video_clips.jsonl",
        [
            {
                "clip_id": "clip-remove",
                "video_id": "video-1",
                "title": "Remove faucet with wrench",
                "description": "Loosen and remove the old faucet.",
                "gemini_metadata": {
                    "clip": {"start_seconds": 10.0, "end_seconds": 20.0}
                },
            },
            {
                "clip_id": "clip-install",
                "video_id": "video-1",
                "title": "Install ceiling fan",
                "description": "Attach the new fan to the ceiling.",
                "gemini_metadata": {
                    "clip": {"start_seconds": 20.0, "end_seconds": 30.0}
                },
            },
        ],
    )
    write_jsonl(
        month1_dir / "action_object_tool_triples.jsonl",
        [
            _triple("clip", "clip-remove", "remove", "faucet", tool="wrench"),
            _triple("clip", "clip-install", "install", "fan"),
        ],
    )
    write_jsonl(month1_dir / "verbnet_mappings.jsonl", [])
    write_jsonl(month2_dir / "framenet_mappings.jsonl", [])
    write_jsonl(month2_dir / "diy_actionnet_v1.jsonl", [])
    return root


def _write_steps(path: Path) -> None:
    write_jsonl(
        path,
        [
            {
                "id": "step-remove",
                "title": "Remove faucet",
                "description": "Loosen the old fixture",
                "tools": ["wrench"],
                "materials": ["plumber putty"],
            },
            {
                "id": "step-paint",
                "title": "Paint wall",
                "description": "Apply the finish",
                "tools": ["roller"],
                "materials": ["paint"],
            },
        ],
    )


def _write_pairs(path: Path) -> None:
    write_jsonl(
        path,
        [
            {
                "id": "comparison-timestamp",
                "step_id": "step-remove",
                "clip_a": {
                    "video_id": "video-1",
                    "start_seconds": 10.0,
                    "end_seconds": 20.0,
                },
                "clip_b": {"clip_id": "clip-install"},
                "winner": "A",
                "winner_clip_id": "clip-remove",
                "judge_provenance": "cascade",
                "judge_confidence": 0.91,
                "judge_id": "cascade-v1",
            },
            {
                "id": "comparison-canonical-ids",
                "step_id": "step-remove",
                "clip_a": {"clip_id": "clip-install"},
                "clip_b": {"clip_id": "clip-remove"},
                "winner": "B",
                "winner_clip_id": "clip-remove",
                "judge_provenance": "human",
                "judge_confidence": 0.75,
                "judge_id": "reviewer-1",
            },
            {
                "id": "comparison-unresolved",
                "step_id": "step-paint",
                "clip_a": {
                    "video_id": "missing-video",
                    "start_seconds": 1.0,
                    "end_seconds": 2.0,
                },
                "clip_b": {"clip_id": "clip-install"},
                "winner": "B",
                "judge_provenance": "cascade",
                "judge_confidence": 0.6,
            },
            {
                "id": "comparison-winner-mismatch",
                "step_id": "step-paint",
                "clip_a": {"clip_id": "clip-remove"},
                "clip_b": {"clip_id": "clip-install"},
                "winner": "A",
                "winner_clip_id": "not-either-clip",
                "judge_provenance": "human",
                "judge_confidence": None,
            },
        ],
    )


def _patch_query_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_parse(query_text: str, *_: Any, **__: Any) -> tuple[list[ActionTriple], None]:
        if query_text.startswith("Remove faucet"):
            return [_triple("step", "query", "remove", "faucet", tool="wrench")], None
        return [_triple("step", "query", "paint", "wall", tool="roller")], None

    monkeypatch.setattr(preference_evaluation, "parse_query_triples", fake_parse)


def test_query_construction_is_deterministic_and_accepts_step_id_alias() -> None:
    step = PreferenceStepInput.model_validate(
        {
            "id": "  step-1  ",
            "title": "  Remove\n faucet ",
            "description": " Loosen   the old faucet ",
            "tools": [" basin   wrench ", "bucket"],
            "materials": [" plumber putty "],
        }
    )

    query, provenance = build_step_query(step)

    assert step.step_id == "step-1"
    assert query == (
        "Remove faucet Loosen the old faucet "
        "Tools: basin wrench, bucket Materials: plumber putty"
    )
    assert provenance == {
        "rule_version": STEP_QUERY_RULE_VERSION,
        "field_order": ["title", "description", "tools", "materials"],
        "title": "Remove faucet",
        "description": "Loosen the old faucet",
        "tools": ["basin wrench", "bucket"],
        "materials": ["plumber putty"],
        "join": "single spaces; inventory prefixes are 'Tools:' and 'Materials:'",
    }


def test_pair_input_accepts_comparison_and_winner_aliases() -> None:
    pair = PreferencePairInput.model_validate(
        {
            "id": " comparison-1 ",
            "step_id": "step-1",
            "clip_a": {"clip_id": "clip-a"},
            "clip_b": {"clip_id": "clip-b"},
            "winner": "B",
            "judge_provenance": " human ",
            "judge_confidence": 0.8,
        }
    )

    assert pair.comparison_id == "comparison-1"
    assert pair.winner_position == "B"
    assert pair.judge_provenance == "human"
    assert pair.judge_confidence == 0.8


@pytest.mark.parametrize(
    ("score_a", "score_b", "judged_winner", "predicted", "credit"),
    [
        (0.9, 0.1, "A", "A", 1.0),
        (0.9, 0.1, "B", "A", 0.0),
        (0.1, 0.9, "B", "B", 1.0),
        (0.1, 0.9, "A", "B", 0.0),
        (0.5, 0.5, "A", "tie", 0.5),
        (0.5, 0.5, "B", "tie", 0.5),
    ],
)
def test_exact_pair_outcome_maps_winner_and_credit(
    score_a: float,
    score_b: float,
    judged_winner: str,
    predicted: str,
    credit: float,
) -> None:
    assert exact_pair_outcome(score_a, score_b, judged_winner) == (predicted, credit)


def test_direct_scoring_uses_production_lexical_structured_and_frozen_hybrid() -> None:
    query = _triple("step", "step-1", "remove", "faucet")
    resources = StructuredResources(
        triples=[
            _triple("clip", "clip-a", "remove", "faucet"),
            _triple("clip", "clip-b", "install", "faucet"),
        ],
        verbnet=[],
        framenet=[],
        taxonomy=[],
    )

    scores = score_query_clip_ids(
        query_triples=[query],
        clip_ids=["clip-a", "clip-b"],
        lexical_scores={"clip-a": 0.2, "clip-b": 0.8},
        resources=resources,
    )

    assert FROZEN_HYBRID_ALPHA == 0.5
    assert scores["clip-a"]["lexical_score"] == 0.2
    assert scores["clip-b"]["lexical_score"] == 0.8
    assert scores["clip-a"]["structured_score"] > scores["clip-b"]["structured_score"]
    for clip_scores in scores.values():
        assert clip_scores["effective_hybrid_alpha_lexical"] == FROZEN_HYBRID_ALPHA
        assert clip_scores["hybrid_score"] == pytest.approx(
            FROZEN_HYBRID_ALPHA * clip_scores["lexical_score"]
            + (1.0 - FROZEN_HYBRID_ALPHA) * clip_scores["structured_score"]
        )


def test_direct_hybrid_scoring_preserves_lexical_fallback_without_action() -> None:
    scores = score_query_clip_ids(
        query_triples=[],
        clip_ids=["clip-a", "clip-b"],
        lexical_scores={"clip-a": 0.2, "clip-b": 0.8},
        resources=StructuredResources(
            triples=[], verbnet=[], framenet=[], taxonomy=[]
        ),
    )

    assert scores["clip-a"]["structured_score"] == 0.0
    assert scores["clip-a"]["hybrid_score"] == 0.2
    assert scores["clip-b"]["hybrid_score"] == 0.8
    assert scores["clip-a"]["effective_hybrid_alpha_lexical"] == 1.0


def test_evaluator_writes_scored_and_retained_unscored_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _write_built_index(tmp_path)
    steps = tmp_path / "steps.synthetic.jsonl"
    pairs = tmp_path / "pairs.synthetic.jsonl"
    output = tmp_path / "evaluation"
    _write_steps(steps)
    _write_pairs(pairs)
    _patch_query_parser(monkeypatch)

    paths = run_preference_evaluation(
        index=index,
        steps_jsonl=steps,
        pairs_jsonl=pairs,
        output_dir=output,
        seed=42,
        bootstrap_iterations=50,
    )

    rows = {row["comparison_id"]: row for row in _read_jsonl(paths["pair_scores"])}
    timestamp = rows["comparison-timestamp"]
    assert timestamp["pair_resolution_status"] == "resolved"
    assert timestamp["original_clip_a_reference"] == {
        "video_id": "video-1",
        "start_seconds": 10.0,
        "end_seconds": 20.0,
    }
    assert timestamp["resolved_canonical_clip_a_id"] == "clip-remove"
    assert timestamp["clip_a_resolution"]["resolution_method"] == "video_timestamp"
    assert timestamp["judged_winner_canonical_clip_id"] == "clip-remove"
    assert timestamp["judge_provenance"] == "cascade"
    assert timestamp["judge_confidence"] == 0.91
    assert timestamp["judge_id"] == "cascade-v1"
    assert timestamp["lexical_predicted_winner"] == "A"
    assert timestamp["structured_predicted_winner"] == "A"
    assert timestamp["hybrid_predicted_winner"] == "A"
    assert timestamp["effective_hybrid_alpha_lexical"] == FROZEN_HYBRID_ALPHA

    canonical = rows["comparison-canonical-ids"]
    assert canonical["pair_resolution_status"] == "resolved"
    assert canonical["clip_a_resolution"]["resolution_method"] == "canonical_clip_id"
    assert canonical["clip_b_resolution"]["resolution_method"] == "canonical_clip_id"
    assert canonical["winner_position"] == "B"
    assert canonical["judged_winner_canonical_clip_id"] == "clip-remove"
    assert canonical["judge_provenance"] == "human"
    assert canonical["judge_confidence"] == 0.75
    assert canonical["lexical_agreement_credit"] == 1.0
    assert canonical["structured_agreement_credit"] == 1.0
    assert canonical["hybrid_agreement_credit"] == 1.0

    unresolved = rows["comparison-unresolved"]
    assert unresolved["pair_resolution_status"] == "unresolved_clip_reference"
    assert unresolved["clip_a_resolution"]["status"] == "unresolved"
    assert unresolved["original_clip_a_reference"]["video_id"] == "missing-video"
    assert unresolved["judge_provenance"] == "cascade"
    assert unresolved["judge_confidence"] == 0.6
    assert unresolved["lexical_a_score"] is None
    assert unresolved["structured_agreement_credit"] is None
    assert unresolved["hybrid_predicted_winner"] is None

    mismatch = rows["comparison-winner-mismatch"]
    assert mismatch["pair_resolution_status"] == "invalid_winner_clip_id"
    assert mismatch["winner_clip_id_input"] == "not-either-clip"
    assert mismatch["judged_winner_canonical_clip_id"] is None
    assert mismatch["lexical_a_score"] is None
    assert mismatch["structured_b_score"] is None
    assert mismatch["hybrid_agreement_credit"] is None

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["counts"]["total_comparisons"] == 4
    assert summary["counts"]["resolved_comparisons"] == 2
    assert summary["counts"]["unresolved_comparisons"] == 2
    assert summary["counts"]["resolution_percentage"] == 50.0
    assert summary["descriptive_counts"]["judge_provenance"] == {
        "cascade": 2,
        "human": 2,
    }
    assert summary["descriptive_counts"]["judge_confidence"] == {
        "present": 3,
        "missing": 1,
        "minimum": 0.6,
        "maximum": 0.91,
        "mean": pytest.approx((0.91 + 0.75 + 0.6) / 3),
    }
    for method in ("lexical", "structured", "hybrid"):
        method_summary = summary["methods"][method]
        assert method_summary["resolved_pairs"] == 2
        assert method_summary["clustered_bootstrap_95_ci"]["cluster_count"] == 1
        assert method_summary["clustered_bootstrap_95_ci"]["cluster_unit"] == "step_id"
        assert method_summary["clustered_bootstrap_95_ci"]["seed"] == 42
        assert method_summary["clustered_bootstrap_95_ci"]["draws"] == 50

    validation = json.loads(paths["validation_report"].read_text(encoding="utf-8"))
    assert validation["status"] == "passed_with_resolution_issues"
    assert validation["resolution_issue_count"] == 2
    assert {issue["code"] for issue in validation["resolution_issues"]} == {
        "unresolved_clip_reference",
        "invalid_winner_clip_id",
    }
    assert validation["comparisons_silently_dropped"] == 0
    assert validation["output_counts"]["pair_score_rows"] == 4
    assert paths["summary_markdown"].is_file()
    assert paths["manifest"].is_file()


def test_malformed_jsonl_fails_with_a_persistent_validation_report(tmp_path: Path) -> None:
    index = tmp_path / "unused-existing-index"
    index.mkdir()
    steps = tmp_path / "steps-malformed.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    output = tmp_path / "failed-evaluation"
    steps.write_text('{"id": "step-1"\n', encoding="utf-8")
    write_jsonl(
        pairs,
        [
            {
                "id": "comparison-1",
                "step_id": "step-1",
                "clip_a": {"clip_id": "clip-a"},
                "clip_b": {"clip_id": "clip-b"},
                "winner": "A",
                "judge_provenance": "cascade",
            }
        ],
    )

    with pytest.raises(ValueError, match="validation failed"):
        run_preference_evaluation(
            index=index,
            steps_jsonl=steps,
            pairs_jsonl=pairs,
            output_dir=output,
            bootstrap_iterations=10,
        )

    report = json.loads(
        (output / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["fatal_error_count"] >= 1
    assert any(error["code"] == "jsonl_read_error" for error in report["fatal_errors"])
    assert any("line 1" in error["message"] for error in report["fatal_errors"])
    assert report["comparisons_silently_dropped"] == 0


def test_evaluate_pairs_cli_passes_frozen_inputs_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "built-index"
    index.mkdir()
    steps = tmp_path / "steps.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    output = tmp_path / "output"
    steps.write_text("{}\n", encoding="utf-8")
    pairs.write_text("{}\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_run_preference_evaluation(**kwargs: Any) -> dict[str, Path]:
        captured.update(kwargs)
        return {
            "summary": output / "summary.json",
            "pair_scores": output / "pair_scores.jsonl",
        }

    monkeypatch.setattr(cli, "run_preference_evaluation", fake_run_preference_evaluation)

    result = CliRunner().invoke(
        cli.app,
        [
            "evaluate-pairs",
            "--index",
            str(index),
            "--steps",
            str(steps),
            "--pairs",
            str(pairs),
            "--output",
            str(output),
            "--seed",
            "42",
            "--bootstrap-iterations",
            "17",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "index": index,
        "steps_jsonl": steps,
        "pairs_jsonl": pairs,
        "output_dir": output,
        "seed": 42,
        "bootstrap_iterations": 17,
    }
    assert "Pairwise preference evaluation complete" in result.output


def test_evaluate_pairs_cli_runs_end_to_end_on_built_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _write_built_index(tmp_path)
    steps = tmp_path / "steps.synthetic.jsonl"
    pairs = tmp_path / "pairs.synthetic.jsonl"
    output = tmp_path / "cli-evaluation"
    _write_steps(steps)
    _write_pairs(pairs)
    _patch_query_parser(monkeypatch)

    result = CliRunner().invoke(
        cli.app,
        [
            "evaluate-pairs",
            "--index",
            str(index),
            "--steps",
            str(steps),
            "--pairs",
            str(pairs),
            "--output",
            str(output),
            "--seed",
            "42",
            "--bootstrap-iterations",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "pair_scores.jsonl").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "summary.md").is_file()
    assert (output / "validation_report.json").is_file()
    assert (output / "manifest.json").is_file()
    assert json.loads((output / "summary.json").read_text())["counts"] == {
        "referenced_steps": 2,
        "resolution_percentage": 50.0,
        "resolution_target_met": False,
        "resolution_target_percentage": 90.0,
        "resolved_comparisons": 2,
        "total_comparisons": 4,
        "unresolved_comparisons": 2,
    }
