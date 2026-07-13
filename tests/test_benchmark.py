import pytest

from action_semantics.models import ActionTriple, ClipRecord
from action_semantics.retrieval.benchmark import (
    RankOutcome,
    _candidate_text,
    _metrics,
    _normalized_phrase_occurs,
    _paired_cluster_bootstrap_delta_cis,
    _rank,
    _select_evaluation_queries,
)


def _query_triple(record_id: str, action: str = "install") -> ActionTriple:
    return ActionTriple(
        record_type="step",
        record_id=record_id,
        source_field="query",
        action=action,
        action_lemma=action,
        action_text=action,
        sentence=f"{action} the fixture",
        extraction_method="test",
    )


def test_candidate_text_uses_shared_clip_fields_only():
    clip = ClipRecord(
        clip_id="clip-1",
        video_id="video-1",
        title="SECRET QUERY TITLE",
        description="Mount the fixture.",
        summary="Finish the installation.",
        gemini_metadata={
            "source_video": {
                "title": "SECRET PARENT TITLE",
                "summary": "SECRET PARENT SUMMARY",
            },
            "clip": {
                "name": "SECRET QUERY TITLE",
                "tools": ["Screwdriver"],
                "supplies": ["Mounting screw"],
            },
        },
    )

    text = _candidate_text(clip)

    assert text == "Mount the fixture. Finish the installation. Screwdriver Mounting screw"
    assert "SECRET" not in text


def test_query_filtering_does_not_change_candidate_pool():
    candidates = [
        ClipRecord(clip_id="eligible", title="Install Faucet", description="Mount fixture"),
        ClipRecord(clip_id="duplicate-a", title="Paint Wall", description="Apply coating"),
        ClipRecord(clip_id="duplicate-b", title="Paint Wall", description="Cover surface"),
        ClipRecord(
            clip_id="leakage",
            title="Clean Sink",
            description="Clean sink thoroughly before use",
        ),
        ClipRecord(clip_id="no-action", title="Project Overview", description="General context"),
    ]
    parsed = {
        "eligible": [_query_triple("eligible")],
        "duplicate-a": [_query_triple("duplicate-a", "paint")],
        "duplicate-b": [_query_triple("duplicate-b", "paint")],
        "leakage": [_query_triple("leakage", "clean")],
    }

    eligible, report = _select_evaluation_queries(candidates, parsed)

    assert [clip.clip_id for clip in eligible] == ["eligible"]
    assert len(candidates) == 5
    assert report["candidate_count"] == 5
    assert report["eligible_query_count"] == 1
    assert report["excluded_query_count"] == 4
    assert report["parseable_action_query_count"] == 4
    assert report["parser_query_coverage"] == 0.8
    assert report["exclusion_counts"] == {
        "ambiguous_normalized_query_title": 2,
        "exact_normalized_query_in_paired_candidate": 1,
        "no_parsed_action": 1,
    }


def test_normalized_phrase_leakage_requires_token_boundaries():
    assert _normalized_phrase_occurs("Clean Sink", "First, CLEAN   sink thoroughly.")
    assert not _normalized_phrase_occurs("mix", "Continue mixing the ingredients.")


def _outcome(rank: int) -> RankOutcome:
    return RankOutcome(
        deterministic_rank=rank,
        best_rank=rank,
        worst_rank=rank,
        top_clip_id="target" if rank == 1 else "other",
        relevant_score=1.0 / rank,
        positive_candidate_count=rank,
        tie_size=1,
    )


def test_rank_records_positive_tie_interval_and_deterministic_display_order():
    outcome = _rank({"correct": 0.5, "wrong": 0.8}, "correct")
    tied = _rank({"z": 0.5, "a": 0.5}, "z")

    assert outcome.deterministic_rank == 2
    assert outcome.best_rank == outcome.worst_rank == 2
    assert outcome.top_clip_id == "wrong"
    assert tied.deterministic_rank == 2
    assert tied.best_rank == 1
    assert tied.worst_rank == 2
    assert tied.tie_size == 2
    assert tied.top_clip_id == "a"


def test_rank_does_not_create_an_arbitrary_rank_from_zero_scores():
    outcome = _rank({"target": 0.0, "other": 0.0}, "target")

    assert outcome.deterministic_rank is None
    assert outcome.best_rank is None
    assert outcome.top_clip_id is None


def test_rank_rejects_missing_relevant_clip():
    with pytest.raises(ValueError, match="Relevant clip is absent"):
        _rank({"candidate": 0.5}, "missing")


def test_benchmark_metrics_report_hits_mrr_and_median():
    metrics = _metrics([_outcome(1), _outcome(2), _outcome(4)])

    assert metrics["hit_at_1"] == 1 / 3
    assert metrics["hit_at_3"] == 2 / 3
    assert metrics["hit_at_10"] == 1.0
    assert metrics["mean_reciprocal_rank"] == (1 + 0.5 + 0.25) / 3
    assert metrics["median_expected_rank_among_positive_targets"] == 2.0


def test_benchmark_metrics_use_expected_tie_credit_and_zero_for_no_evidence():
    tied = RankOutcome(
        deterministic_rank=2,
        best_rank=1,
        worst_rank=2,
        top_clip_id="a",
        relevant_score=0.5,
        positive_candidate_count=2,
        tie_size=2,
    )
    no_evidence = RankOutcome(
        deterministic_rank=None,
        best_rank=None,
        worst_rank=None,
        top_clip_id=None,
        relevant_score=0.0,
        positive_candidate_count=0,
        tie_size=0,
    )

    metrics = _metrics([tied, no_evidence])

    assert metrics["target_positive_score_rate"] == 0.5
    assert metrics["positive_target_tie_rate"] == 0.5
    assert metrics["hit_at_1"] == 0.25
    assert metrics["hit_at_3"] == 0.5
    assert metrics["mean_reciprocal_rank"] == pytest.approx(0.375)


def test_empty_metrics_are_explicitly_unavailable():
    metrics = _metrics([])

    assert metrics["query_count"] == 0
    assert metrics["hit_at_1"] is None
    assert metrics["mean_reciprocal_rank"] is None


def test_cluster_bootstrap_is_paired_and_deterministic():
    arguments = {
        "video_ids": ["video-a", "video-a", "video-b"],
        "baseline_ranks": [_outcome(2), _outcome(4), _outcome(1)],
        "challenger_ranks": [_outcome(1), _outcome(2), _outcome(1)],
        "iterations": 200,
        "seed": 99,
    }

    first = _paired_cluster_bootstrap_delta_cis(**arguments)
    second = _paired_cluster_bootstrap_delta_cis(**arguments)

    assert first == second
    assert first["cluster_unit"] == "video"
    assert first["cluster_count"] == 2
    assert first["metrics"]["hit_at_1"]["estimate"] == pytest.approx(1 / 3)
    assert first["metrics"]["mean_reciprocal_rank"]["estimate"] == pytest.approx(
        ((1 + 0.5 + 1) - (0.5 + 0.25 + 1)) / 3
    )
    for result in first["metrics"].values():
        assert result["ci_95_low"] <= result["estimate"] <= result["ci_95_high"]
