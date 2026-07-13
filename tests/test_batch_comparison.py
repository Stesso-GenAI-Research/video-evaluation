import csv
import json
from pathlib import Path

import pytest

from action_semantics.retrieval import batch_comparison
from action_semantics.retrieval.scorers import STRUCTURED_SCORER_VERSION


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _clip(clip_id: str, video_id: str) -> dict:
    return {
        "clip_id": clip_id,
        "video_id": video_id,
        "url": f"https://example.test/{video_id}",
        "title": f"Title {clip_id}",
        "gemini_metadata": {
            "clip": {
                "segment_id": f"segment-{clip_id}",
                "start_seconds": 10.0,
                "end_seconds": 20.0,
            },
            "source_video": {"title": f"Video {video_id}"},
        },
    }


def _write_retrieval_resources(month1_dir: Path, month2_dir: Path) -> None:
    month1_dir.mkdir(parents=True, exist_ok=True)
    month2_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        month1_dir / "action_object_tool_triples.jsonl",
        month1_dir / "verbnet_mappings.jsonl",
        month2_dir / "framenet_mappings.jsonl",
        month2_dir / "diy_actionnet_v1.jsonl",
    ):
        path.write_text(f'{{"artifact":"{path.name}"}}\n', encoding="utf-8")


def _make_scoring_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    _write_retrieval_resources(tmp_path / "month1", tmp_path / "month2")
    clips_path = tmp_path / "score-clips.jsonl"
    _write_jsonl(
        clips_path,
        [_clip("a", "v1"), _clip("b", "v2"), _clip("c", "v3"), _clip("d", "v4")],
    )
    input_path = tmp_path / "score-input.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "step_id": "step-1",
                "query": "remove faucet",
                "original_matches": [
                    {"clip_id": "a", "rank": 1},
                    {"clip_id": "b", "rank": 2},
                ],
            },
            {
                "step_id": "step-2",
                "query": "install light",
                "original_matches": [
                    {"clip_id": "c", "rank": 1},
                    {"clip_id": "d", "rank": 2},
                ],
            },
        ],
    )

    def fake_rank(**kwargs: object) -> dict:
        ids = ["b", "c"] if kwargs["query_text"] == "remove faucet" else ["a", "d"]
        return {"results": [{"clip_id": clip_id} for clip_id in ids]}

    monkeypatch.setattr(batch_comparison, "rank_indexed_clips", fake_rank)
    return batch_comparison.run_batch_comparison(
        comparisons_jsonl=input_path,
        clips_jsonl=clips_path,
        month1_dir=tmp_path / "month1",
        month2_dir=tmp_path / "month2",
        output_dir=tmp_path / "score-output",
        spacy_model="test-model",
        challenger_method="hybrid",
        top_k=2,
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_run_batch_comparison_writes_neutral_and_blinded_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips_path = tmp_path / "clips.jsonl"
    _write_jsonl(
        clips_path,
        [_clip("a", "v1"), _clip("b", "v2"), _clip("c", "v3"), _clip("d", "v4")],
    )
    input_path = tmp_path / "comparisons.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "step_id": "step-1",
                "query": "remove faucet",
                "original_matches": [
                    {"clip_id": "c", "rank": 1},
                    {"clip_id": "a", "rank": 2},
                ],
            },
            {
                "step_id": "step-2",
                "query": "install light",
                "original_matches": [
                    {"clip_id": "b", "rank": 1},
                    {"clip_id": "d", "rank": 2},
                ],
            },
        ],
    )
    calls: list[dict] = []

    def fake_rank(**kwargs: object) -> dict:
        calls.append(kwargs)
        query = str(kwargs["query_text"])
        ids = ["b", "c"] if query == "remove faucet" else ["d"]
        return {"results": [{"clip_id": clip_id} for clip_id in ids]}

    monkeypatch.setattr(batch_comparison, "rank_indexed_clips", fake_rank)
    _write_retrieval_resources(tmp_path / "month1", tmp_path / "month2")
    first = batch_comparison.run_batch_comparison(
        comparisons_jsonl=input_path,
        clips_jsonl=clips_path,
        month1_dir=tmp_path / "month1",
        month2_dir=tmp_path / "month2",
        output_dir=tmp_path / "output-1",
        spacy_model="test-model",
        challenger_method="hybrid",
        top_k=2,
        hybrid_alpha=0.4,
    )
    second = batch_comparison.run_batch_comparison(
        comparisons_jsonl=input_path,
        clips_jsonl=clips_path,
        month1_dir=tmp_path / "month1",
        month2_dir=tmp_path / "month2",
        output_dir=tmp_path / "output-2",
        spacy_model="test-model",
        challenger_method="hybrid",
        top_k=2,
        hybrid_alpha=0.4,
    )

    assert len(calls) == 4
    assert calls[0]["method"] == "hybrid"
    assert calls[0]["top_k"] == 2
    assert calls[0]["hybrid_alpha"] == 0.4

    rankings = [json.loads(line) for line in first["rankings"].read_text().splitlines()]
    assert [row["step_id"] for row in rankings] == ["step-1", "step-2"]
    assert [row["clip_id"] for row in rankings[0]["original_matches"]] == ["c", "a"]
    assert [row["rank"] for row in rankings[0]["original_matches"]] == [1, 2]
    assert [row["clip_id"] for row in rankings[0]["challenger"]["matches"]] == [
        "b",
        "c",
    ]
    assert rankings[0]["set_comparison"]["overlap_count"] == 1
    assert rankings[0]["set_comparison"]["jaccard"] == pytest.approx(1 / 3)
    assert rankings[0]["quality_claim"] is False
    assert rankings[0]["winner"] is None
    assert rankings[0]["configuration"] == {
        "challenger_method": "hybrid",
        "requested_top_k": 2,
        "hybrid_alpha_lexical": 0.4,
        "spacy_model": "test-model",
    }
    assert rankings[0]["provenance"]["structured_scorer"] == (
        STRUCTURED_SCORER_VERSION
    )
    assert rankings[0]["provenance"]["taxonomy_used_for_ranking"] is False
    assert rankings[0]["provenance"]["taxonomy_used_for_diagnostics"] is True

    summary = json.loads(first["summary"].read_text())
    assert summary["counts"] == {
        "steps": 2,
        "original_results": 4,
        "challenger_results": 3,
        "blind_review_candidates": 5,
    }
    assert summary["coverage"]["steps_with_any_challenger_results"] == 2
    assert summary["coverage"]["steps_with_full_challenger_top_k"] == 1
    assert summary["coverage"]["challenger_slot_coverage"] == 0.75
    assert summary["overlap"]["total_shared_clips"] == 2
    assert summary["jaccard"]["mean"] == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert summary["quality_claim"] is False
    assert summary["winner"] is None
    assert summary["configuration"] == rankings[0]["configuration"]
    assert summary["provenance"] == rankings[0]["provenance"]

    with first["blind_review"].open(newline="", encoding="utf-8") as handle:
        review = list(csv.DictReader(handle))
        headers = handle.seek(0) or next(csv.reader(handle))
    assert len(review) == 5
    assert len({(row["step_id"], row["clip_id"]) for row in review}) == 5
    assert all(row["overall_relevant"] == "" for row in review)
    assert all(row["action"] == "" for row in review)
    assert all(row["object"] == "" for row in review)
    assert all(row["tool"] == "" for row in review)
    assert "method" not in headers
    assert "original" not in headers
    assert "challenger" not in headers

    assert first["rankings"].read_bytes() == second["rankings"].read_bytes()
    assert first["summary"].read_bytes() == second["summary"].read_bytes()
    assert first["blind_review"].read_bytes() == second["blind_review"].read_bytes()


@pytest.mark.parametrize(
    "rows, error",
    [
        (
            [
                {
                    "step_id": "same",
                    "query": "first",
                    "original_matches": [{"clip_id": "a", "rank": 1}],
                },
                {
                    "step_id": "same",
                    "query": "second",
                    "original_matches": [{"clip_id": "b", "rank": 1}],
                },
            ],
            "step_id values must be unique",
        ),
        (
            [
                {
                    "step_id": "duplicate-clips",
                    "query": "query",
                    "original_matches": [
                        {"clip_id": "a", "rank": 1},
                        {"clip_id": "a", "rank": 2},
                    ],
                }
            ],
            "duplicate clip_id",
        ),
        (
            [
                {
                    "step_id": "bad-ranks",
                    "query": "query",
                    "original_matches": [
                        {"clip_id": "a", "rank": 2},
                        {"clip_id": "b", "rank": 1},
                    ],
                }
            ],
            "contiguous rank order",
        ),
        (
            [
                {
                    "step_id": "wrong-type",
                    "query": "query",
                    "original_matches": [{"clip_id": "a", "rank": "1"}],
                }
            ],
            "valid integer",
        ),
        (
            [
                {
                    "step_id": "extra",
                    "query": "query",
                    "original_matches": [{"clip_id": "a", "rank": 1}],
                    "unexpected": True,
                }
            ],
            "Extra inputs are not permitted",
        ),
    ],
)
def test_batch_comparison_rejects_invalid_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict],
    error: str,
) -> None:
    clips_path = tmp_path / "clips.jsonl"
    _write_jsonl(clips_path, [_clip("a", "v1"), _clip("b", "v2")])
    input_path = tmp_path / "comparisons.jsonl"
    _write_jsonl(input_path, rows)

    def should_not_search(**_: object) -> dict:
        raise AssertionError("search must not run before validation succeeds")

    monkeypatch.setattr(batch_comparison, "rank_indexed_clips", should_not_search)
    with pytest.raises(ValueError, match=error):
        batch_comparison.run_batch_comparison(
            comparisons_jsonl=input_path,
            clips_jsonl=clips_path,
            month1_dir=tmp_path / "month1",
            month2_dir=tmp_path / "month2",
            output_dir=tmp_path / "output",
            spacy_model="test-model",
        )
    assert not (tmp_path / "output").exists()


def test_batch_comparison_rejects_unknown_original_clip_before_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips_path = tmp_path / "clips.jsonl"
    _write_jsonl(clips_path, [_clip("a", "v1")])
    input_path = tmp_path / "comparisons.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "step_id": "step-1",
                "query": "query",
                "original_matches": [{"clip_id": "missing", "rank": 1}],
            }
        ],
    )

    def should_not_search(**_: object) -> dict:
        raise AssertionError("search must not run when an original ID is unknown")

    monkeypatch.setattr(batch_comparison, "rank_indexed_clips", should_not_search)
    with pytest.raises(ValueError, match="not in the indexed corpus"):
        batch_comparison.run_batch_comparison(
            comparisons_jsonl=input_path,
            clips_jsonl=clips_path,
            month1_dir=tmp_path / "month1",
            month2_dir=tmp_path / "month2",
            output_dir=tmp_path / "output",
            spacy_model="test-model",
        )
    assert not (tmp_path / "output").exists()


def test_batch_comparison_requires_same_top_k_for_original_and_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips_path = tmp_path / "clips.jsonl"
    _write_jsonl(clips_path, [_clip("a", "v1"), _clip("b", "v2")])
    input_path = tmp_path / "comparisons.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "step_id": "step-1",
                "query": "query",
                "original_matches": [{"clip_id": "a", "rank": 1}],
            }
        ],
    )
    def should_not_search(**_: object) -> dict:
        raise AssertionError("search must not run")

    monkeypatch.setattr(batch_comparison, "rank_indexed_clips", should_not_search)

    with pytest.raises(ValueError, match="exactly top_k=2"):
        batch_comparison.run_batch_comparison(
            comparisons_jsonl=input_path,
            clips_jsonl=clips_path,
            month1_dir=tmp_path / "month1",
            month2_dir=tmp_path / "month2",
            output_dir=tmp_path / "output",
            spacy_model="test-model",
            top_k=2,
        )


def test_score_batch_review_uses_complete_steps_and_exact_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_scoring_artifacts(tmp_path, monkeypatch)
    fieldnames, rows = _read_csv(paths["blind_review"])
    overall_labels = {
        ("step-1", "a"): "YES",
        ("step-1", "b"): "0",
        ("step-1", "c"): "true",
        ("step-2", "c"): "n",
        ("step-2", "d"): "false",
        ("step-2", "a"): "1",
    }
    action_labels = {
        ("step-1", "a"): "y",
        ("step-1", "b"): "yes",
        ("step-1", "c"): "no",
    }
    for row in rows:
        key = (row["step_id"], row["clip_id"])
        row["overall_relevant"] = overall_labels[key]
        row["action"] = action_labels.get(key, "")
        row["object"] = "yes" if key == ("step-1", "a") else ""
    _write_csv(paths["blind_review"], fieldnames, rows)

    output_path = tmp_path / "review-scores.json"
    report = batch_comparison.score_batch_review(
        paths["rankings"], paths["blind_review"], output_path
    )

    overall_coverage = report["judgment_coverage"]["dimensions"]["overall_relevant"]
    assert overall_coverage == {
        "labeled_rows": 6,
        "total_rows": 6,
        "row_coverage": 1.0,
        "complete_steps": 2,
        "partial_steps": 0,
        "unlabeled_steps": 0,
        "total_steps": 2,
        "complete_step_coverage": 1.0,
    }
    overall = report["dimensions"]["overall_relevant"]
    assert overall["original"]["mean_precision_at_k"] == 0.25
    assert overall["challenger"]["mean_precision_at_k"] == 0.5
    assert overall["delta_challenger_minus_original"]["mean_precision_at_k"] == 0.25
    assert overall["original"]["success_at_k"] == 0.5
    assert overall["challenger"]["success_at_k"] == 1.0
    assert overall["precision_wins_ties_losses"] == {
        "challenger_wins": 1,
        "ties": 1,
        "challenger_losses": 0,
    }
    assert overall["success_wins_ties_losses"] == {
        "challenger_wins": 1,
        "ties": 1,
        "challenger_losses": 0,
    }
    bootstrap = overall["paired_bootstrap_95_ci_mean_precision_delta"]
    assert bootstrap["n"] == 2
    assert bootstrap["estimate"] == 0.25
    assert bootstrap["lower"] == 0.0
    assert bootstrap["upper"] == 0.5

    action_coverage = report["judgment_coverage"]["dimensions"]["action"]
    assert action_coverage["complete_steps"] == 1
    assert action_coverage["unlabeled_steps"] == 1
    action = report["dimensions"]["action"]
    assert action["original"]["mean_precision_at_k"] == 1.0
    assert action["challenger"]["mean_precision_at_k"] == 0.5
    assert action["precision_wins_ties_losses"]["challenger_losses"] == 1

    object_coverage = report["judgment_coverage"]["dimensions"]["object"]
    assert object_coverage["partial_steps"] == 1
    assert object_coverage["unlabeled_steps"] == 1
    assert report["dimensions"]["object"]["scored_steps"] == 0
    assert report["dimensions"]["object"]["original"]["mean_precision_at_k"] is None
    assert report["dimensions"]["tool"]["scored_steps"] == 0
    assert json.loads(output_path.read_text()) == report

    alternate = batch_comparison.score_batch_review(
        paths["rankings"],
        paths["blind_review"],
        tmp_path / "alternate-bootstrap.json",
        seed=7,
    )
    assert alternate["blind_review_seed"] == 1729
    assert alternate["bootstrap_seed"] == 7
    assert (
        alternate["dimensions"]["overall_relevant"]
        ["paired_bootstrap_95_ci_mean_precision_delta"]["seed"]
        == 7
    )


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("duplicate_review_id", "Review IDs must be unique"),
        ("wrong_ab_rank", "A/B ranks inconsistent"),
        ("missing_row", "missing rows"),
        ("invalid_label", "Invalid overall_relevant label"),
    ],
)
def test_score_batch_review_validates_blind_worksheet_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error: str,
) -> None:
    paths = _make_scoring_artifacts(tmp_path, monkeypatch)
    fieldnames, rows = _read_csv(paths["blind_review"])
    if mutation == "duplicate_review_id":
        rows[1]["review_id"] = rows[0]["review_id"]
    elif mutation == "wrong_ab_rank":
        row = next(row for row in rows if row["set_a_rank"])
        row["set_a_rank"] = "99"
    elif mutation == "missing_row":
        rows.pop()
    elif mutation == "invalid_label":
        rows[0]["overall_relevant"] = "maybe"
    _write_csv(paths["blind_review"], fieldnames, rows)

    output_path = tmp_path / "invalid-review-scores.json"
    with pytest.raises(ValueError, match=error):
        batch_comparison.score_batch_review(
            paths["rankings"], paths["blind_review"], output_path
        )
    assert not output_path.exists()


def test_score_batch_review_validates_seeded_blind_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_scoring_artifacts(tmp_path, monkeypatch)
    rankings = [json.loads(line) for line in paths["rankings"].read_text().splitlines()]
    assignment = rankings[0]["blind_review_assignment"]
    assignment["original_set"], assignment["challenger_set"] = (
        assignment["challenger_set"],
        assignment["original_set"],
    )
    _write_jsonl(paths["rankings"], rankings)

    with pytest.raises(ValueError, match="inconsistent with seed"):
        batch_comparison.score_batch_review(
            paths["rankings"],
            paths["blind_review"],
            tmp_path / "invalid-assignment.json",
        )
