from __future__ import annotations

import pytest
from pydantic import ValidationError

from action_semantics.models import ClipRecord
from action_semantics.retrieval.clip_resolution import (
    DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
    ClipReference,
    ClipResolver,
)


def _clip(
    clip_id: str,
    video_id: str,
    *,
    start_seconds: float = 10.0,
    end_seconds: float = 20.0,
) -> ClipRecord:
    return ClipRecord.model_validate(
        {
            "clip_id": clip_id,
            "video_id": video_id,
            "gemini_metadata": {
                "clip": {
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                }
            },
        }
    )


def test_resolver_resolves_exact_canonical_id_and_is_serializable() -> None:
    resolver = ClipResolver([_clip("clip-1", "video-1")])
    reference = ClipReference(clip_id="clip-1")

    result = resolver.resolve(reference)

    assert result.status == "resolved"
    assert result.is_resolved is True
    assert result.canonical_clip_id == "clip-1"
    assert result.resolution_method == "canonical_clip_id"
    assert result.model_dump(mode="json") == {
        "original_reference": {"clip_id": "clip-1"},
        "status": "resolved",
        "canonical_clip_id": "clip-1",
        "resolution_method": "canonical_clip_id",
        "candidate_clip_ids": ["clip-1"],
        "message": None,
    }


def test_resolver_resolves_timestamps_with_default_tolerance() -> None:
    resolver = ClipResolver([_clip("clip-1", "video-1")])
    reference = ClipReference(
        video_id="video-1",
        start_seconds=10.04,
        end_seconds=19.96,
    )

    result = resolver.resolve(reference)

    assert DEFAULT_TIMESTAMP_TOLERANCE_SECONDS == 0.05
    assert result.status == "resolved"
    assert result.canonical_clip_id == "clip-1"
    assert result.resolution_method == "video_timestamp"
    assert result.original_reference == {
        "video_id": "video-1",
        "start_seconds": 10.04,
        "end_seconds": 19.96,
    }


def test_resolver_returns_explicit_unresolved_timestamp_result() -> None:
    resolver = ClipResolver([_clip("clip-1", "video-1")])

    result = resolver.resolve(
        ClipReference(
            video_id="video-1",
            start_seconds=11.0,
            end_seconds=21.0,
        )
    )

    assert result.status == "unresolved"
    assert result.is_resolved is False
    assert result.canonical_clip_id is None
    assert result.candidate_clip_ids == ()
    assert result.message == (
        "no canonical clip matches video_id='video-1', start=11.0, end=21.0 "
        "within ±0.05 seconds"
    )


def test_resolver_returns_all_candidates_for_ambiguous_timestamps() -> None:
    resolver = ClipResolver(
        [
            _clip("clip-z", "video-1", start_seconds=10.01, end_seconds=20.01),
            _clip("clip-a", "video-1", start_seconds=9.99, end_seconds=19.99),
            _clip("other-video", "video-2"),
        ],
        tolerance_seconds=0.02,
    )

    result = resolver.resolve(
        ClipReference(
            video_id="video-1",
            start_seconds=10.0,
            end_seconds=20.0,
        )
    )

    assert result.status == "ambiguous"
    assert result.canonical_clip_id is None
    assert result.candidate_clip_ids == ("clip-a", "clip-z")
    assert result.message == (
        "timestamp reference is ambiguous for video_id='video-1', start=10.0, "
        "end=20.0: ['clip-a', 'clip-z']"
    )


def test_resolver_returns_conflict_when_id_and_timestamps_disagree() -> None:
    resolver = ClipResolver([_clip("clip-1", "video-1")])
    reference = ClipReference(
        clip_id="clip-1",
        video_id="video-2",
        start_seconds=10.0,
        end_seconds=20.0,
    )

    result = resolver.resolve(reference)

    assert result.status == "conflict"
    assert result.canonical_clip_id is None
    assert result.candidate_clip_ids == ("clip-1",)
    assert result.original_reference == {
        "clip_id": "clip-1",
        "video_id": "video-2",
        "start_seconds": 10.0,
        "end_seconds": 20.0,
    }
    assert result.message == (
        "clip_id 'clip-1' conflicts with its supplied video/timestamps"
    )


def test_unknown_id_does_not_silently_fall_back_to_matching_timestamps() -> None:
    resolver = ClipResolver([_clip("canonical-a", "video-1")])

    result = resolver.resolve(
        ClipReference(
            clip_id="stale-id",
            video_id="video-1",
            start_seconds=10.0,
            end_seconds=20.0,
        )
    )

    assert result.status == "conflict"
    assert result.canonical_clip_id is None
    assert result.candidate_clip_ids == ("canonical-a",)
    assert "unknown canonical clip_id" in str(result.message)


def test_legacy_mode_can_fall_back_from_unknown_id_to_matching_timestamps() -> None:
    resolver = ClipResolver(
        [_clip("canonical-a", "video-1")],
        allow_unknown_clip_id_timestamp_fallback=True,
    )

    result = resolver.resolve(
        ClipReference(
            clip_id="legacy-id",
            video_id="video-1",
            start_seconds=10.0,
            end_seconds=20.0,
        )
    )

    assert result.status == "resolved"
    assert result.canonical_clip_id == "canonical-a"
    assert result.resolution_method == "video_timestamp"


def test_clip_reference_is_strict_and_requires_complete_timestamps() -> None:
    with pytest.raises(ValidationError, match="valid number"):
        ClipReference(
            video_id="video-1",
            start_seconds="10.0",
            end_seconds=20.0,
        )
    with pytest.raises(ValidationError, match="must be supplied together"):
        ClipReference(clip_id="clip-1", video_id="video-1", start_seconds=10.0)
