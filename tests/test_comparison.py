import pytest

from action_semantics.io_utils import sha256_file, write_jsonl
from action_semantics.models import ClipRecord
from action_semantics.retrieval.comparison import compare_result_sets
from action_semantics.retrieval.lexical import tfidf_scores
from action_semantics.retrieval.scorers import STRUCTURED_SCORER_VERSION


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


def _write_retrieval_resources(directory):
    for filename in (
        "action_object_tool_triples.jsonl",
        "verbnet_mappings.jsonl",
        "framenet_mappings.jsonl",
        "diy_actionnet_v1.jsonl",
    ):
        (directory / filename).write_text(f'{{"artifact":"{filename}"}}\n')


def test_comparison_names_generated_reference_as_lexical_not_original(
    tmp_path, monkeypatch
):
    clips_path = tmp_path / "clips.jsonl"
    write_jsonl(clips_path, [{"clip_id": value} for value in ["a", "b", "c"]])
    _write_retrieval_resources(tmp_path)

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
    assert result["reference"]["results"][0]["ranking_method"] == "lexical"
    assert result["challenger"]["results"][0]["ranking_method"] == "hybrid"
    assert result["set_difference"]["overlap_clip_ids"] == ["b"]
    assert result["set_difference"]["jaccard"] == 1 / 3
    assert result["quality_claim"] is False
    assert result["winner"] is None
    assert result["configuration"] == {
        "challenger_method": "hybrid",
        "hybrid_alpha_lexical": 0.5,
        "spacy_model": "unused",
    }
    assert result["provenance"]["structured_scorer"] == STRUCTURED_SCORER_VERSION
    assert result["provenance"]["artifacts"]["clips"]["sha256"] == sha256_file(
        clips_path
    )
    assert result["provenance"]["taxonomy_used_for_ranking"] is False
    assert result["provenance"]["taxonomy_used_for_diagnostics"] is True


def test_comparison_preserves_explicit_original_order_and_checks_ids(
    tmp_path, monkeypatch
):
    clips_path = tmp_path / "clips.jsonl"
    write_jsonl(clips_path, [{"clip_id": value} for value in ["a", "b", "c"]])
    _write_retrieval_resources(tmp_path)
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
    assert all(row["score"] is None for row in result["reference"]["results"])
    with pytest.raises(ValueError, match="not in the indexed corpus"):
        compare_result_sets(
            query_text="remove faucet",
            clips_jsonl=clips_path,
            month1_dir=tmp_path,
            month2_dir=tmp_path,
            spacy_model="unused",
            original_clip_ids=["missing"],
        )


def test_comparison_does_not_turn_zero_score_ties_into_arbitrary_results(
    tmp_path, monkeypatch
):
    clips_path = tmp_path / "clips.jsonl"
    write_jsonl(clips_path, [{"clip_id": "a"}, {"clip_id": "b"}])
    _write_retrieval_resources(tmp_path)
    monkeypatch.setattr(
        "action_semantics.retrieval.comparison.rank_indexed_clips",
        lambda **kwargs: {
            "results": [
                {"clip_id": "a", "score": 0.0},
                {"clip_id": "b", "score": 0.0},
            ],
            "warnings": [],
        },
    )

    result = compare_result_sets(
        query_text="zzzzqqq",
        clips_jsonl=clips_path,
        month1_dir=tmp_path,
        month2_dir=tmp_path,
        spacy_model="unused",
        top_k=2,
    )

    assert result["reference"]["results"] == []
    assert result["challenger"]["results"] == []
    assert result["set_difference"]["overlap_count"] == 0
    assert len(result["warnings"]) == 2
