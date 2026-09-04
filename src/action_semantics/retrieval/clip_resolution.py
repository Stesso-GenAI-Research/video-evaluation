"""Resolve external clip references to canonical indexed clips.

References may name a canonical ``clip_id``, a source-video time interval, or
both.  Resolution itself is non-throwing once a reference has passed input
validation: callers receive an explicit status and can choose whether an
unresolved row should be reported, retained, or treated as fatal.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from action_semantics.models import ClipRecord


DEFAULT_TIMESTAMP_TOLERANCE_SECONDS: Final[float] = 0.05

ClipResolutionStatus = Literal["resolved", "unresolved", "ambiguous", "conflict"]
ClipResolutionMethod = Literal["canonical_clip_id", "video_timestamp"]


class ClipReference(BaseModel):
    """Strict external reference to one canonical clip.

    A canonical ID alone is sufficient.  Timestamp lookup requires all three
    of ``video_id``, ``start_seconds``, and ``end_seconds``.  Supplying both
    forms allows the resolver to verify that they agree.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    clip_id: str | None = None
    video_id: str | int | None = None
    start_seconds: float | int | None = None
    end_seconds: float | int | None = None

    @field_validator("clip_id")
    @classmethod
    def clip_id_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("clip_id must not be blank")
        return value.strip() if value is not None else None

    @field_validator("video_id")
    @classmethod
    def video_id_is_not_blank(cls, value: str | int | None) -> str | int | None:
        if isinstance(value, str) and not value.strip():
            raise ValueError("video_id must not be blank")
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def has_complete_reference(self) -> "ClipReference":
        timestamp_values = (self.video_id, self.start_seconds, self.end_seconds)
        has_any_timestamp = any(value is not None for value in timestamp_values)
        has_all_timestamps = all(value is not None for value in timestamp_values)
        if self.clip_id is None and not has_all_timestamps:
            raise ValueError(
                "provide clip_id or video_id + start_seconds + end_seconds"
            )
        if has_any_timestamp and not has_all_timestamps:
            raise ValueError(
                "video_id, start_seconds, and end_seconds must be supplied together"
            )
        if has_all_timestamps and float(self.start_seconds) < 0.0:
            raise ValueError("start_seconds must be non-negative")
        if has_all_timestamps and float(self.end_seconds) <= float(self.start_seconds):
            raise ValueError("end_seconds must be greater than start_seconds")
        return self

    def reference_key(self) -> tuple[Any, ...]:
        """Return a stable key suitable for duplicate-reference validation."""
        if self.clip_id is not None:
            return ("clip_id", self.clip_id)
        return (
            "timestamp",
            str(self.video_id),
            float(self.start_seconds),
            float(self.end_seconds),
        )

    def reference_dict(self) -> dict[str, Any]:
        """Return the normalized supplied fields without subclass-only data."""
        return {
            key: value
            for key, value in {
                "clip_id": self.clip_id,
                "video_id": self.video_id,
                "start_seconds": self.start_seconds,
                "end_seconds": self.end_seconds,
            }.items()
            if value is not None
        }


class ClipResolutionResult(BaseModel):
    """Serializable outcome of resolving one :class:`ClipReference`."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    original_reference: dict[str, Any]
    status: ClipResolutionStatus
    canonical_clip_id: str | None = None
    resolution_method: ClipResolutionMethod | None = None
    candidate_clip_ids: tuple[str, ...] = ()
    message: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved"


def clip_interval(clip: ClipRecord) -> tuple[float | None, float | None]:
    """Read a canonical clip interval from its indexed-video metadata."""
    metadata = clip.gemini_metadata.get("clip", {})
    if not isinstance(metadata, dict):
        return None, None
    start = metadata.get("start_seconds")
    end = metadata.get("end_seconds")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None, None
    return float(start), float(end)


class ClipResolver:
    """Reusable canonical-ID and per-video timestamp lookup for a clip corpus."""

    def __init__(
        self,
        clips: Iterable[ClipRecord],
        *,
        tolerance_seconds: float = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
        allow_unknown_clip_id_timestamp_fallback: bool = False,
    ) -> None:
        if tolerance_seconds < 0.0:
            raise ValueError("tolerance_seconds must be non-negative")

        clips_by_id: dict[str, ClipRecord] = {}
        clips_by_video: defaultdict[str, list[ClipRecord]] = defaultdict(list)
        duplicate_ids: set[str] = set()
        for clip in clips:
            if clip.clip_id in clips_by_id:
                duplicate_ids.add(clip.clip_id)
            else:
                clips_by_id[clip.clip_id] = clip
            clips_by_video[str(clip.video_id)].append(clip)
        if duplicate_ids:
            raise ValueError(
                f"clips contains duplicate clip_id values: {sorted(duplicate_ids)}"
            )

        self.tolerance_seconds = float(tolerance_seconds)
        self.allow_unknown_clip_id_timestamp_fallback = (
            allow_unknown_clip_id_timestamp_fallback
        )
        self._clips_by_id = clips_by_id
        self._clips_by_video = dict(clips_by_video)

    def resolve(self, reference: ClipReference) -> ClipResolutionResult:
        """Resolve ``reference`` without dropping information on a failure."""
        original_reference = reference.reference_dict()

        if reference.clip_id is not None and reference.clip_id in self._clips_by_id:
            if reference.video_id is not None:
                clip = self._clips_by_id[reference.clip_id]
                start, end = clip_interval(clip)
                if (
                    str(clip.video_id) != str(reference.video_id)
                    or start is None
                    or end is None
                    or abs(start - float(reference.start_seconds)) > self.tolerance_seconds
                    or abs(end - float(reference.end_seconds)) > self.tolerance_seconds
                ):
                    message = (
                        f"clip_id {reference.clip_id!r} conflicts with its supplied "
                        "video/timestamps"
                    )
                    return ClipResolutionResult(
                        original_reference=original_reference,
                        status="conflict",
                        candidate_clip_ids=(reference.clip_id,),
                        message=message,
                    )
            return ClipResolutionResult(
                original_reference=original_reference,
                status="resolved",
                canonical_clip_id=reference.clip_id,
                resolution_method="canonical_clip_id",
                candidate_clip_ids=(reference.clip_id,),
            )

        if reference.video_id is None:
            message = f"unknown canonical clip_id: {reference.clip_id!r}"
            return ClipResolutionResult(
                original_reference=original_reference,
                status="unresolved",
                message=message,
            )

        target_start = float(reference.start_seconds)
        target_end = float(reference.end_seconds)
        candidates: list[str] = []
        for clip in self._clips_by_video.get(str(reference.video_id), []):
            start, end = clip_interval(clip)
            if (
                start is not None
                and end is not None
                and abs(start - target_start) <= self.tolerance_seconds
                and abs(end - target_end) <= self.tolerance_seconds
            ):
                candidates.append(clip.clip_id)

        sorted_candidates = tuple(sorted(candidates))
        if (
            reference.clip_id is not None
            and not self.allow_unknown_clip_id_timestamp_fallback
        ):
            message = (
                f"unknown canonical clip_id {reference.clip_id!r} conflicts with its "
                "supplied video/timestamps"
            )
            return ClipResolutionResult(
                original_reference=original_reference,
                status="conflict",
                candidate_clip_ids=sorted_candidates,
                message=message,
            )
        if not candidates:
            message = (
                "no canonical clip matches "
                f"video_id={reference.video_id!r}, start={target_start}, "
                f"end={target_end} within ±{self.tolerance_seconds} seconds"
            )
            return ClipResolutionResult(
                original_reference=original_reference,
                status="unresolved",
                message=message,
            )
        if len(candidates) > 1:
            message = (
                "timestamp reference is ambiguous for "
                f"video_id={reference.video_id!r}, start={target_start}, end={target_end}: "
                f"{list(sorted_candidates)}"
            )
            return ClipResolutionResult(
                original_reference=original_reference,
                status="ambiguous",
                candidate_clip_ids=sorted_candidates,
                message=message,
            )
        return ClipResolutionResult(
            original_reference=original_reference,
            status="resolved",
            canonical_clip_id=candidates[0],
            resolution_method="video_timestamp",
            candidate_clip_ids=(candidates[0],),
        )
