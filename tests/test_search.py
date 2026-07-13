from action_semantics.models import (
    ActionTriple,
    FrameNetMapping,
    TaxonomyAssignment,
    VerbNetMapping,
)
from action_semantics.io_utils import write_jsonl
from action_semantics.retrieval.scorers import StructuredResources, structured_score
from action_semantics.retrieval.search import QUERY_ID, rank_indexed_clips


def _triple(
    record_type: str,
    record_id: str,
    action: str,
    obj: str,
    *,
    tool: str | None = None,
    negated: bool = False,
) -> ActionTriple:
    return ActionTriple(
        record_type=record_type,
        record_id=record_id,
        source_field="test",
        action=action,
        action_lemma=action,
        action_text=action,
        object_lemmas=[obj],
        tool_lemmas=[tool] if tool else [],
        negated=negated,
        sentence=f"{action} the {obj}",
        extraction_method="test",
    )


def test_structured_search_prefers_matching_action_and_object():
    triples = [
        _triple("step", "query", "remove", "faucet"),
        _triple("clip", "good", "remove", "faucet"),
        _triple("clip", "wrong_action", "install", "faucet"),
    ]
    resources = StructuredResources(
        triples=triples,
        verbnet=[
            VerbNetMapping(action_lemma="remove", verbnet_classes=["remove-10.1"], has_mapping=True),
            VerbNetMapping(action_lemma="install", verbnet_classes=["put-9.1"], has_mapping=True),
        ],
        framenet=[
            FrameNetMapping(action_lemma="remove", frames=["Removing"], has_mapping=True),
            FrameNetMapping(action_lemma="install", frames=["Placing"], has_mapping=True),
        ],
        taxonomy=[
            TaxonomyAssignment(action_lemma="remove", cluster_id=1, cluster_label="remove", support_count=2),
            TaxonomyAssignment(action_lemma="install", cluster_id=2, cluster_label="install", support_count=1),
        ],
    )

    good = structured_score("query", "good", resources)["structured_score"]
    wrong = structured_score("query", "wrong_action", resources)["structured_score"]

    assert good > wrong


def test_score_does_not_join_object_from_an_unrelated_action():
    triples = [
        _triple("step", "query", "remove", "faucet"),
        _triple("clip", "aligned", "remove", "faucet"),
        _triple("clip", "split", "remove", "filter"),
        _triple("clip", "split", "paint", "faucet"),
    ]
    resources = StructuredResources(triples=triples, verbnet=[], framenet=[], taxonomy=[])

    aligned = structured_score("query", "aligned", resources)
    split = structured_score("query", "split", resources)

    assert aligned["structured_score"] > split["structured_score"]
    assert split["object_match"] == 0.0


def test_negated_action_does_not_match_positive_query():
    triples = [
        _triple("step", "query", "remove", "faucet"),
        _triple("clip", "positive", "remove", "faucet"),
        _triple("clip", "negated", "remove", "faucet", negated=True),
    ]
    resources = StructuredResources(triples=triples, verbnet=[], framenet=[], taxonomy=[])

    assert structured_score("query", "positive", resources)["structured_score"] > 0
    assert structured_score("query", "negated", resources)["structured_score"] == 0


def test_taxonomy_is_diagnostic_and_cannot_make_unrelated_actions_match():
    triples = [
        _triple("step", "query", "remove", "faucet"),
        _triple("clip", "unrelated", "install", "faucet"),
    ]
    resources = StructuredResources(
        triples=triples,
        verbnet=[],
        framenet=[],
        taxonomy=[
            TaxonomyAssignment(
                action_lemma="remove", cluster_id=1, cluster_label="mixed", support_count=2
            ),
            TaxonomyAssignment(
                action_lemma="install", cluster_id=1, cluster_label="mixed", support_count=2
            ),
        ],
    )

    result = structured_score("query", "unrelated", resources)

    assert result["taxonomy_match"] == 1.0
    assert result["structured_score"] == 0.0


def test_public_search_ranks_results_and_applies_video_diversity(tmp_path, monkeypatch):
    clips_path = tmp_path / "clips.jsonl"
    write_jsonl(
        clips_path,
        [
            {"clip_id": "best", "video_id": "v1", "title": "Remove faucet"},
            {"clip_id": "same-video", "video_id": "v1", "title": "Remove filter"},
            {"clip_id": "other-video", "video_id": "v2", "title": "Install faucet"},
        ],
    )
    query = _triple("step", QUERY_ID, "remove", "faucet")
    resources = StructuredResources(
        triples=[
            _triple("clip", "best", "remove", "faucet"),
            _triple("clip", "same-video", "remove", "filter"),
            _triple("clip", "other-video", "install", "faucet"),
        ],
        verbnet=[],
        framenet=[],
        taxonomy=[],
    )
    monkeypatch.setattr(
        "action_semantics.retrieval.search._query_triples", lambda *_: [query]
    )
    monkeypatch.setattr(
        "action_semantics.retrieval.search.resources_from_files", lambda *_: resources
    )
    monkeypatch.setattr(
        "action_semantics.retrieval.search.tfidf_scores",
        lambda *_: {"best": 0.9, "same-video": 0.8, "other-video": 0.7},
    )

    result = rank_indexed_clips(
        query_text="remove faucet",
        clips_jsonl=clips_path,
        month1_dir=tmp_path,
        month2_dir=tmp_path,
        spacy_model="unused",
        method="hybrid",
        top_k=2,
        max_per_video=1,
    )

    assert result["schema_version"] == "search.v2"
    assert [row["clip_id"] for row in result["results"]] == ["best", "other-video"]
    assert [row["rank"] for row in result["results"]] == [1, 2]
    assert result["index"]["taxonomy_used_for_ranking"] is False


def test_hybrid_search_falls_back_to_lexical_when_query_parse_fails(
    tmp_path, monkeypatch
):
    clips_path = tmp_path / "clips.jsonl"
    write_jsonl(clips_path, [{"clip_id": "clip-1", "title": "Faucet removal"}])
    monkeypatch.setattr("action_semantics.retrieval.search._query_triples", lambda *_: [])
    monkeypatch.setattr(
        "action_semantics.retrieval.search.tfidf_scores", lambda *_: {"clip-1": 0.8}
    )

    result = rank_indexed_clips(
        query_text="faucet removal",
        clips_jsonl=clips_path,
        month1_dir=tmp_path,
        month2_dir=tmp_path,
        spacy_model="unused",
    )

    assert result["method"] == "lexical_fallback"
    assert result["results"][0]["clip_id"] == "clip-1"
    assert "no verb" in result["warnings"][0]
