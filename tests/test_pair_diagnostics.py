from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from action_semantics.io_utils import write_jsonl
from action_semantics.pair_diagnostics import analyze_pair_scores


def _resolved_row(
    comparison_id: str,
    *,
    lexical: tuple[str, float],
    structured: tuple[str, float],
    hybrid: tuple[str, float],
    pair_type: str,
) -> dict:
    row = {
        "comparison_id": comparison_id,
        "step_id": f"step-{comparison_id}",
        "pair_resolution_status": "resolved",
        "query_text": "install the panel",
        "query_diagnostics": {"missing_action_parse": False},
        "judge_provenance": "synthetic_test",
        "judge_confidence": 0.5,
        "winner_position": "A",
        "resolved_canonical_clip_a_id": f"{comparison_id}-a",
        "resolved_canonical_clip_b_id": f"{comparison_id}-b",
        "input_metadata": {"synthetic_pair_type": pair_type},
        "structured_a_diagnostics": {
            "action_match": 1.0,
            "object_match": 1.0,
            "context_match": 0.0,
            "missing_structured_evidence": False,
        },
        "structured_b_diagnostics": {
            "action_match": 0.0,
            "object_match": 0.0,
            "context_match": 0.0,
            "missing_structured_evidence": True,
        },
    }
    for method, (prediction, credit) in {
        "lexical": lexical,
        "structured": structured,
        "hybrid": hybrid,
    }.items():
        if prediction == "A":
            score_a, score_b = 0.8, 0.2
        elif prediction == "B":
            score_a, score_b = 0.2, 0.8
        else:
            score_a = score_b = 0.0
        row[f"{method}_a_score"] = score_a
        row[f"{method}_b_score"] = score_b
        row[f"{method}_predicted_winner"] = prediction
        row[f"{method}_agreement_credit"] = credit
    return row


def test_analyze_pair_scores_writes_prioritized_descriptive_outputs(
    tmp_path: Path,
) -> None:
    pair_scores = tmp_path / "pair_scores.jsonl"
    write_jsonl(
        pair_scores,
        [
            _resolved_row(
                "one",
                lexical=("B", 0.0),
                structured=("A", 1.0),
                hybrid=("A", 1.0),
                pair_type="action-contrast",
            ),
            _resolved_row(
                "two",
                lexical=("A", 1.0),
                structured=("tie", 0.5),
                hybrid=("A", 1.0),
                pair_type="within-video",
            ),
        ],
    )

    paths = analyze_pair_scores(
        pair_scores_jsonl=pair_scores,
        output_dir=tmp_path / "diagnostics",
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["counts"] == {
        "total_comparisons": 2,
        "resolved_comparisons": 2,
        "unresolved_comparisons": 0,
    }
    assert summary["diagnostic_flag_counts"]["structured_only_correct"] == 1
    assert summary["diagnostic_flag_counts"]["structured_tie"] == 1
    assert set(summary["pair_type_slices"]) == {"action-contrast", "within-video"}
    assert summary["changes_scoring_configuration"] is False
    with paths["cases"].open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    assert cases[0]["comparison_id"] == "one"
    assert "structured_only_correct" in cases[0]["diagnostic_flags"]
    assert paths["markdown"].is_file()
    assert paths["manifest"].is_file()


def test_analyze_pair_scores_rejects_missing_fields(tmp_path: Path) -> None:
    pair_scores = tmp_path / "bad.jsonl"
    write_jsonl(pair_scores, [{"comparison_id": "incomplete"}])
    with pytest.raises(ValueError, match="missing fields"):
        analyze_pair_scores(
            pair_scores_jsonl=pair_scores,
            output_dir=tmp_path / "diagnostics",
        )
