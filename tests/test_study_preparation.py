import csv
import json

import pytest

from action_semantics.io_utils import write_csv, write_jsonl
from action_semantics.retrieval.preference_evaluation import PreferencePairInput, PreferenceStepInput
from action_semantics.study_preparation import (
    ANSWER_FIELDS, _evidence_group, check_prospective_holdout, export_reviewed_pairs,
)


def _sources(tmp_path, development, candidate):
    a, b = tmp_path / "development.jsonl", tmp_path / "candidate.jsonl"
    write_jsonl(a, development)
    write_jsonl(b, candidate)
    return dict(development_source=a, candidate_source=b, output_dir=tmp_path / "check")


def test_new_database_id_cannot_hide_same_youtube_video(tmp_path):
    args = _sources(tmp_path,
                    [{"video_id": 1, "youtube_id": "abc", "url": "https://youtube.com/watch?v=abc"}],
                    [{"video_id": 999, "url": "https://youtu.be/abc?t=30"}])
    result = check_prospective_holdout(**args)
    assert result["status"] == "fail"
    assert result["overlapping_candidate_rows"] == 1


def test_holdout_rejects_duplicates_even_when_source_is_new(tmp_path):
    args = _sources(tmp_path, [{"video_id": 1}],
                    [{"video_id": 2, "youtube_id": "new"},
                     {"video_id": 3, "url": "https://www.youtube.com/shorts/new"}])
    result = check_prospective_holdout(**args)
    assert result["status"] == "fail"
    assert result["overlapping_candidate_rows"] == 0
    assert result["duplicate_candidate_rows"] == 1


def test_disjoint_sources_only_pass_identity_check(tmp_path):
    result = check_prospective_holdout(**_sources(tmp_path, [{"video_id": 1}], [{"video_id": 2}]))
    assert result["status"] == "pass_identity_check_only"
    assert result["limitations"]


def _kit(tmp_path, answers):
    kit = tmp_path / "kit"
    (kit / "private").mkdir(parents=True)
    rows = [{"review_id": f"r-{i}", "step_id": "step-1", "query": "Remove faucet",
             "a_clip_id": "clip-1", "b_clip_id": "clip-2", "preference": answer,
             "confidence": "", "notes": ""} for i, answer in enumerate(answers)]
    key = {"rows": {r["review_id"]: {"immutable": {k: v for k, v in r.items() if k not in ANSWER_FIELDS}}
                    for r in rows},
           "steps": {"step-1": {"id": "step-1", "title": "Remove faucet", "description": "",
                                   "tools": [], "materials": []}}}
    (kit / "private/review_key.json").write_text(json.dumps(key))
    worksheet = kit / "preference_review.csv"
    write_csv(worksheet, rows)
    return dict(kit=kit, worksheet=worksheet, output_dir=tmp_path / "export", judge="test-reviewer")


def test_export_does_not_invent_preferences_for_blank_reviews(tmp_path):
    args = _kit(tmp_path, ["", "tie", "unjudgeable"])
    with pytest.raises(ValueError, match="No A/B judgments"):
        export_reviewed_pairs(**args)
    assert not args["output_dir"].exists()


def test_export_preserves_unscored_cases_and_generates_valid_contracts(tmp_path):
    args = _kit(tmp_path, ["A", "B", "tie", "unjudgeable", ""])
    summary = export_reviewed_pairs(**args)
    assert summary["exported_binary_judgments"] == 2
    assert summary["excluded_rows"] == 3
    assert summary["held_out"] is False
    pairs = [PreferencePairInput.model_validate(json.loads(line)) for line in
             (args["output_dir"] / "pairwise.jsonl").read_text().splitlines()]
    assert [p.winner_position for p in pairs] == ["A", "B"]
    assert all(p.judge_provenance == "human_diagnostic_pilot" for p in pairs)
    for line in (args["output_dir"] / "steps.jsonl").read_text().splitlines():
        PreferenceStepInput.model_validate(json.loads(line))
    exclusions = [json.loads(line) for line in
                  (args["output_dir"] / "excluded_judgments.jsonl").read_text().splitlines()]
    assert {r["reason"] for r in exclusions} == {"tie", "unjudgeable", "pending"}


@pytest.mark.parametrize("change", ["swap", "drop", "duplicate", "nan"])
def test_export_rejects_changed_ordering_missing_rows_and_invalid_confidence(tmp_path, change):
    args = _kit(tmp_path, ["A", "B"])
    with args["worksheet"].open() as stream:
        rows = list(csv.DictReader(stream))
    if change == "swap":
        rows[0]["a_clip_id"], rows[0]["b_clip_id"] = rows[0]["b_clip_id"], rows[0]["a_clip_id"]
    elif change == "drop":
        rows.pop()
    elif change == "duplicate":
        rows[1] = rows[0]
    else:
        rows[0]["confidence"] = "NaN"
    write_csv(args["worksheet"], rows)
    with pytest.raises(ValueError):
        export_reviewed_pairs(**args)


def test_zero_evidence_analysis_distinguishes_missing_parses_from_zero_matching():
    row = {"structured_a_score": 0, "structured_b_score": 0,
           "query_diagnostics": {"missing_action_parse": False},
           "structured_a_diagnostics": {"candidate_action_count": 2, "action_match": 0},
           "structured_b_diagnostics": {"candidate_action_count": 3, "action_match": 0}}
    assert _evidence_group(row) == "both_candidates_parsed_but_zero_action_match"
    row["query_diagnostics"]["missing_action_parse"] = True
    assert _evidence_group(row) == "query_has_no_parsed_action"
