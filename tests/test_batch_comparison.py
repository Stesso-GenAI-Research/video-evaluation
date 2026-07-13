import csv
import json
from pathlib import Path

import pytest

from action_semantics.retrieval import batch_comparison


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
