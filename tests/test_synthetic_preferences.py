from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from action_semantics.io_utils import write_jsonl
from action_semantics.retrieval.preference_evaluation import (
    PreferencePairInput,
    PreferenceStepInput,
)
from action_semantics.synthetic_preferences import (
    generate_controlled_preferences,
    generate_synthetic_preferences,
)


def _built_index(tmp_path: Path) -> Path:
    root = tmp_path / "built-index"
    input_dir = root / "input"
    month1_dir = root / "month1"
    month2_dir = root / "month2"
    input_dir.mkdir(parents=True)
    month1_dir.mkdir()
    month2_dir.mkdir()
    clips = []
    triples = []
    for video_number in range(1, 4):
        for clip_number in range(2):
            start = float(clip_number * 10)
            end = start + 10.0
            clip_id = f"clip-{video_number}-{clip_number}"
            action = "install" if clip_number == 0 else "remove"
            clips.append(
                {
                    "clip_id": clip_id,
                    "video_id": f"video-{video_number}",
                    "title": f"{action.title()} Panel {video_number}",
                    "description": f"{action.title()} panel {video_number} with care.",
                    "gemini_metadata": {
                        "source_video": {"category": {"name": "Synthetic test"}},
                        "clip": {
                            "start_seconds": start,
                            "end_seconds": end,
                            "tools": [f"Tool {video_number}"],
                            "supplies": [f"Supply {clip_number}"],
                        },
                    },
                }
            )
            triples.append(
                {
                    "record_type": "clip",
                    "record_id": clip_id,
                    "source_field": "title",
                    "action": action.title(),
                    "action_lemma": action,
                    "action_text": action.title(),
                    "object_text": f"Panel {video_number}",
                    "object_lemmas": ["panel"],
                    "sentence": f"{action.title()} Panel {video_number}",
                    "extraction_method": "synthetic_test_fixture",
                    "confidence": 1.0,
                }
            )
    write_jsonl(input_dir / "indexed_video_clips.jsonl", clips)
    write_jsonl(month1_dir / "action_object_tool_triples.jsonl", triples)
    write_jsonl(month1_dir / "verbnet_mappings.jsonl", [])
    write_jsonl(month2_dir / "framenet_mappings.jsonl", [])
    write_jsonl(month2_dir / "diy_actionnet_v1.jsonl", [])
    return root


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generator_writes_valid_balanced_and_auditable_pseudo_judgments(
    tmp_path: Path,
) -> None:
    index = _built_index(tmp_path)
    output = tmp_path / "synthetic-preferences"

    paths = generate_synthetic_preferences(
        index=index,
        output_dir=output,
        step_count=3,
        seed=42,
    )

    steps = _read_jsonl(paths["steps"])
    pairs = _read_jsonl(paths["pairs"])
    assert len(steps) == 3
    assert len(pairs) == 6
    assert all(PreferenceStepInput.model_validate(row) for row in steps)
    assert all(PreferencePairInput.model_validate(row) for row in pairs)
    assert Counter(row["step_id"] for row in pairs) == {
        row["id"]: 2 for row in steps
    }
    assert Counter(row["winner_position"] for row in pairs) == {"A": 3, "B": 3}
    assert {row["synthetic_pair_type"] for row in pairs} == {
        "within-video",
        "cross-video",
    }
    assert all(row["synthetic"] is True for row in pairs)
    assert all(row["judge_confidence"] == 0.5 for row in pairs)
    assert any("video_id" in row["clip_a"] for row in pairs)
    assert any("clip_id" in row["clip_a"] for row in pairs)

    with paths["audit"].open(newline="", encoding="utf-8") as handle:
        audit = list(csv.DictReader(handle))
    assert len(audit) == 6
    assert all(row["step_title"] and row["synthetic_label_rationale"] for row in audit)
    assert "not" in paths["notice"].read_text(encoding="utf-8").lower()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["parameters"]["research_evidence"] is False
    assert manifest["parameters"]["pair_count"] == 6


def test_generator_is_deterministic_and_protects_existing_outputs(tmp_path: Path) -> None:
    index = _built_index(tmp_path)
    first = generate_synthetic_preferences(
        index=index,
        output_dir=tmp_path / "first",
        step_count=3,
        seed=7,
    )
    second = generate_synthetic_preferences(
        index=index,
        output_dir=tmp_path / "second",
        step_count=3,
        seed=7,
    )

    for key in ("steps", "pairs", "audit"):
        assert first[key].read_text(encoding="utf-8") == second[key].read_text(
            encoding="utf-8"
        )
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        generate_synthetic_preferences(
            index=index,
            output_dir=tmp_path / "first",
            step_count=3,
            seed=7,
        )


def test_controlled_generator_writes_hard_negative_development_pairs(
    tmp_path: Path,
) -> None:
    index = _built_index(tmp_path)
    paths = generate_controlled_preferences(
        index=index,
        output_dir=tmp_path / "controlled",
        step_count=2,
        seed=11,
    )

    steps = _read_jsonl(paths["steps"])
    pairs = _read_jsonl(paths["pairs"])
    assert len(steps) == 2
    assert len(pairs) >= 6
    assert all(PreferenceStepInput.model_validate(row) for row in steps)
    assert all(PreferencePairInput.model_validate(row) for row in pairs)
    assert {row["step_id"] for row in pairs} == {row["id"] for row in steps}
    assert {row["winner_position"] for row in pairs} == {"A", "B"}
    assert "within-video-adjacent" in {
        row["synthetic_pair_type"] for row in pairs
    }
    assert "lexical-hard-negative" in {
        row["synthetic_pair_type"] for row in pairs
    }
    assert all(row["development_only"] is True for row in pairs)
    assert all("no human review" in row["adjudication_basis"] for row in pairs)
    notice = paths["notice"].read_text(encoding="utf-8")
    assert "not\nhuman preference data" in notice
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["parameters"]["human_judgments"] is False
    assert manifest["parameters"]["selection_uses_frozen_scores"] is True
