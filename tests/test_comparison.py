import pytest

from action_semantics.io_utils import write_jsonl
from action_semantics.models import ClipRecord
from action_semantics.retrieval.comparison import compare_result_sets
from action_semantics.retrieval.lexical import tfidf_scores


def test_lexical_baseline_prefers_matching_clip_text():
    clips = [
        ClipRecord(clip_id="faucet", title="Remove faucet handle"),
        ClipRecord(clip_id="paint", title="Paint a bedroom wall"),
    ]

    scores = tfidf_scores("remove old faucet", clips)

    assert scores["faucet"] > scores["paint"]


def _search_result(method, ids):
    return {
        "method": method,
        "results": [
            {
                "rank": rank,
                "clip_id": clip_id,
                "score": 1.0 / rank,
                "signals": {"lexical": 0.0, "structured": 0.0},
            }
            for rank, clip_id in enumerate(ids, start=1)
        ],
        "warnings": [],
    }


def test_comparison_names_generated_reference_as_lexical_not_original(
    tmp_path, monkeypatch
):
    clips_path = tmp_path / "clips.jsonl"
    write_jsonl(clips_path, [{"clip_id": value} for value in ["a", "b", "c"]])

    def fake_search(**kwargs):
        if kwargs["method"] == "lexical":
            return _search_result("lexical", ["a", "b", "c"])
        return _search_result("hybrid", ["b", "c", "a"])

    monkeypatch.setattr(
        "action_semantics.retrieval.comparison.rank_indexed_clips", fake_search
    )

    result = compare_result_sets(
        query_text="remove faucet",
        clips_jsonl=clips_path,
        month1_dir=tmp_path,
        month2_dir=tmp_path,
        spacy_model="unused",
        top_k=2,
    )

    assert result["reference"]["label"] == "lexical_baseline"
    assert "original" not in result["reference"]["label"]
    assert result["set_difference"]["overlap_clip_ids"] == ["b"]
    assert result["set_difference"]["jaccard"] == 1 / 3
    assert result["quality_claim"] is False
    assert result["winner"] is None


def test_comparison_preserves_explicit_original_order_and_checks_ids(
    tmp_path, monkeypatch
):
    clips_path = tmp_path / "clips.jsonl"
    write_jsonl(clips_path, [{"clip_id": value} for value in ["a", "b", "c"]])
    monkeypatch.setattr(
        "action_semantics.retrieval.comparison.rank_indexed_clips",
        lambda **kwargs: _search_result(kwargs["method"], ["a", "b", "c"]),
    )

    result = compare_result_sets(
        query_text="remove faucet",
        clips_jsonl=clips_path,
        month1_dir=tmp_path,
        month2_dir=tmp_path,
        spacy_model="unused",
        original_clip_ids=["c", "a"],
        top_k=2,
    )

    assert result["reference"]["label"] == "provided_original"
    assert [row["clip_id"] for row in result["reference"]["results"]] == ["c", "a"]
    with pytest.raises(ValueError, match="not in the indexed corpus"):
        compare_result_sets(
            query_text="remove faucet",
            clips_jsonl=clips_path,
            month1_dir=tmp_path,
            month2_dir=tmp_path,
            spacy_model="unused",
            original_clip_ids=["missing"],
        )
