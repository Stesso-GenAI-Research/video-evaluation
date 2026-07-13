"""Validate and canonicalize the nested IndexedVideo JSONL export.

The source file has one row per video and a nested ``clips`` list.  A source
video can contain repeated entries for the same timestamped segment, including
exact duplicates and variants with richer tool or supply text.  This module
turns those rows into one stable search record per valid video segment while
retaining the original variants for auditability.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .io_utils import iter_jsonl, sha256_file, write_jsonl
from .text import normalize_text


class _StrictSourceModel(BaseModel):
    """Base class that makes source-schema changes visible immediately."""

    model_config = ConfigDict(extra="forbid")


class IndexedVideoCategory(_StrictSourceModel):
    id: int | None = None
    name: str


class IndexedClipSource(_StrictSourceModel):
    name: str
    description: str = ""
    goal: str = ""
    tools: str | list[str] = ""
    supplies: str | list[str] = ""
    start: Decimal
    end: Decimal

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        if not normalize_text(value):
            raise ValueError("clip name is blank")
        return value


class IndexedVideoSource(_StrictSourceModel):
    video_id: int
    youtube_id: str = ""
    source: str = ""
    url: str = ""
    category: IndexedVideoCategory | None = None
    title: str = ""
    summary: str = ""
    goal: str = ""
    views: int | None = None
    likes: int | None = None
    comment_count: int | None = None
    subscribers: int | None = None
    clip_count: int | None = None
    clips: list[IndexedClipSource] = Field(default_factory=list)
    # ``iter_jsonl`` adds this provenance value before validation.
    source_row_sha256: str


class ParsedInventoryItem(_StrictSourceModel):
    """A conservative parse of one tool/supply annotation."""

    name: str
    alternatives: list[str] = Field(default_factory=list)
    purpose: str | None = None
    raw: str


@dataclass(frozen=True)
class _ClipOccurrence:
    video: IndexedVideoSource
    clip: IndexedClipSource
    source_row_number: int
    source_clip_index: int


@dataclass(frozen=True)
class _CanonicalizationResult:
    clips: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    profile: dict[str, Any]


def _parse_inventory_items(value: Any) -> list[ParsedInventoryItem]:
    """Parse inventory annotations while retaining every original fragment."""
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        text = str(value)
        if " used for " in text.lower() or " alternatives:" in text.lower():
            values = re.split(r"\.\s*,\s*(?=[A-Z0-9])", text)
        else:
            values = text.split(",")
    output: list[ParsedInventoryItem] = []
    for item in values:
        raw = normalize_text(item).rstrip(".")
        if not raw:
            continue
        primary = re.split(
            r"\s+alternatives:|\s+used for", raw, maxsplit=1, flags=re.I
        )[0]
        primary = re.sub(r"\s+(unknown|not specified)$", "", primary, flags=re.I)
        alternatives_match = re.search(
            r"\s+alternatives:\s*(.*?)(?=\s+used for\s+|$)", raw, flags=re.I
        )
        alternatives = (
            _ordered_unique(alternatives_match.group(1).split(","))
            if alternatives_match
            else []
        )
        purpose_match = re.search(r"\s+used for\s+(.*)$", raw, flags=re.I)
        purpose = normalize_text(purpose_match.group(1)).rstrip(".") if purpose_match else None
        if cleaned := normalize_text(primary).rstrip("."):
            output.append(
                ParsedInventoryItem(
                    name=cleaned,
                    alternatives=alternatives,
                    purpose=purpose or None,
                    raw=raw,
                )
            )
    return output


def _as_string_list(value: Any) -> list[str]:
    """Compatibility view containing primary inventory item names only."""
    return [item.name for item in _parse_inventory_items(value)]


def _ordered_unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_text(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            output.append(cleaned)
            seen.add(key)
    return output


def _ordered_unique_inventory(items: Iterable[ParsedInventoryItem]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        data = item.model_dump(mode="json")
        signature = json.dumps(data, sort_keys=True, ensure_ascii=False).casefold()
        if signature not in seen:
            output.append(data)
            seen.add(signature)
    return output


def _decimal_text(value: Decimal) -> str:
    """Return the same token for numerically equivalent timestamp values."""
    if not value:
        return "0"
    return format(value.normalize(), "f")


def _timestamp_token(value: Decimal) -> str:
    return _decimal_text(value).replace("-", "m").replace(".", "p")


def _segment_id(video_id: int, start: Decimal, end: Decimal) -> str:
    return (
        f"indexed-video-{video_id}-segment-"
        f"{_timestamp_token(start)}-{_timestamp_token(end)}"
    )


def _source_clip_dict(occurrence: _ClipOccurrence) -> dict[str, Any]:
    clip = occurrence.clip
    return {
        "source_row_number": occurrence.source_row_number,
        "source_clip_index": occurrence.source_clip_index,
        "name": clip.name,
        "description": clip.description,
        "goal": clip.goal,
        "tools": clip.tools,
        "supplies": clip.supplies,
        "start": float(clip.start),
        "end": float(clip.end),
    }


def _source_video_dict(video: IndexedVideoSource) -> dict[str, Any]:
    category = video.category
    return {
        "video_id": str(video.video_id),
        "youtube_id": normalize_text(video.youtube_id) or None,
        "source": normalize_text(video.source) or None,
        "url": normalize_text(video.url) or None,
        "category": (
            {"id": category.id, "name": normalize_text(category.name)}
            if category is not None
            else None
        ),
        "title": normalize_text(video.title) or None,
        "summary": normalize_text(video.summary) or None,
        "goal": normalize_text(video.goal) or None,
        "views": video.views,
        "likes": video.likes,
        "comment_count": video.comment_count,
        "subscribers": video.subscribers,
        "declared_clip_count": video.clip_count,
    }


def _variant_signature(occurrence: _ClipOccurrence) -> str:
    payload = _source_clip_dict(occurrence)
    payload.pop("source_row_number")
    payload.pop("source_clip_index")
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _canonical_clip(occurrences: list[_ClipOccurrence]) -> dict[str, Any]:
    """Build one clip from an encounter-ordered timestamp group."""
    representative = occurrences[0]
    video = representative.video
    clip = representative.clip
    names = _ordered_unique(item.clip.name for item in occurrences)
    descriptions = _ordered_unique(item.clip.description for item in occurrences)
    goals = _ordered_unique(item.clip.goal for item in occurrences)
    tools = _ordered_unique(
        tool for item in occurrences for tool in _as_string_list(item.clip.tools)
    )
    supplies = _ordered_unique(
        supply for item in occurrences for supply in _as_string_list(item.clip.supplies)
    )
    tool_items = _ordered_unique_inventory(
        parsed
        for item in occurrences
        for parsed in _parse_inventory_items(item.clip.tools)
    )
    supply_items = _ordered_unique_inventory(
        parsed
        for item in occurrences
        for parsed in _parse_inventory_items(item.clip.supplies)
    )
    raw_tools = _ordered_unique(
        str(item.clip.tools) for item in occurrences if normalize_text(item.clip.tools)
    )
    raw_supplies = _ordered_unique(
        str(item.clip.supplies) for item in occurrences if normalize_text(item.clip.supplies)
    )
    signatures = {_variant_signature(item) for item in occurrences}
    canonical_description = descriptions[0] if descriptions else None
    canonical_goal = goals[0] if goals else None

    return {
        "clip_id": _segment_id(video.video_id, clip.start, clip.end),
        "video_id": str(video.video_id),
        "url": normalize_text(video.url) or None,
        "title": normalize_text(clip.name),
        # Use the first nonempty value across timestamp-identical variants so
        # a sparse duplicate cannot hide richer searchable text.
        "description": canonical_description,
        # ClipRecord uses ``summary`` for the segment-level goal.  The exact
        # source field name is also retained below in structured metadata.
        "summary": canonical_goal,
        "gemini_metadata": {
            "source_video": _source_video_dict(video),
            "clip": {
                "representative_source_row_number": representative.source_row_number,
                "representative_source_clip_index": representative.source_clip_index,
                "source_clip_indices": [item.source_clip_index for item in occurrences],
                "start_seconds": float(clip.start),
                "end_seconds": float(clip.end),
                "name": normalize_text(clip.name),
                "description": canonical_description,
                "goal": canonical_goal,
                "names": names,
                "aliases": names[1:],
                "description_variants": descriptions,
                "goal_variants": goals,
                "tools": tools,
                "supplies": supplies,
                "tool_items": tool_items,
                "supply_items": supply_items,
                # Singular keys remain for compatibility with existing
                # reports; plural keys preserve every raw source variant.
                "source_tools_text": normalize_text(clip.tools) or None,
                "source_supplies_text": normalize_text(clip.supplies) or None,
                "source_tools_text_variants": raw_tools,
                "source_supplies_text_variants": raw_supplies,
                "variant_count": len(occurrences),
                "merge_kind": (
                    "single"
                    if len(occurrences) == 1
                    else "exact_duplicates"
                    if len(signatures) == 1
                    else "metadata_variants"
                ),
                "source_variants": [_source_clip_dict(item) for item in occurrences],
            },
        },
        "dense_embeddings": {},
        "source_row_sha256": video.source_row_sha256,
    }


def _parse_source_row(row: dict[str, Any], row_number: int) -> IndexedVideoSource:
    try:
        return IndexedVideoSource.model_validate(row)
    except ValidationError as exc:
        raise ValueError(
            f"Indexed video row {row_number} does not match the expected source schema:\n{exc}"
        ) from exc


def _canonicalize_indexed_videos(source_path: Path) -> _CanonicalizationResult:
    groups: dict[tuple[int, Decimal, Decimal], list[_ClipOccurrence]] = {}
    rejected: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    raw_coverage: Counter[str] = Counter()
    video_ids: list[int] = []
    raw_clip_count = 0
    declared_count_mismatches = 0

    for row_number, row in enumerate(iter_jsonl(source_path), start=1):
        video = _parse_source_row(row, row_number)
        video_ids.append(video.video_id)
        if video.clip_count is not None and video.clip_count != len(video.clips):
            declared_count_mismatches += 1
        if video.category is not None and (category_name := normalize_text(video.category.name)):
            category_counts[category_name] += 1

        for clip_index, clip in enumerate(video.clips):
            raw_clip_count += 1
            raw_coverage["clip_description"] += bool(normalize_text(clip.description))
            raw_coverage["clip_goal"] += bool(normalize_text(clip.goal))
            raw_coverage["clip_tools"] += bool(_as_string_list(clip.tools))
            raw_coverage["clip_supplies"] += bool(_as_string_list(clip.supplies))
            occurrence = _ClipOccurrence(video, clip, row_number, clip_index)
            if clip.end <= clip.start:
                reason = "non_positive_duration"
                rejection_reasons[reason] += 1
                rejected.append(
                    {
                        "reason": reason,
                        "video_id": str(video.video_id),
                        "source_row_sha256": video.source_row_sha256,
                        **_source_clip_dict(occurrence),
                    }
                )
                continue
            groups.setdefault((video.video_id, clip.start, clip.end), []).append(occurrence)

    if raw_clip_count == 0:
        raise ValueError(f"{source_path} did not contain any nested clip records")

    clips = [_canonical_clip(occurrences) for occurrences in groups.values()]
    duplicate_groups = [occurrences for occurrences in groups.values() if len(occurrences) > 1]
    exact_duplicate_groups = [
        occurrences
        for occurrences in duplicate_groups
        if len({_variant_signature(item) for item in occurrences}) == 1
    ]
    alias_groups = [
        occurrences
        for occurrences in groups.values()
        if len(_ordered_unique(item.clip.name for item in occurrences)) > 1
    ]
    duplicate_video_id_count = sum(
        count - 1 for count in Counter(video_ids).values() if count > 1
    )

    coverage_counts = Counter()
    for clip in clips:
        metadata = clip["gemini_metadata"]["clip"]
        coverage_counts["clip_description"] += bool(clip["description"])
        coverage_counts["clip_goal"] += bool(clip["summary"])
        coverage_counts["clip_tools"] += bool(metadata["tools"])
        coverage_counts["clip_supplies"] += bool(metadata["supplies"])

    canonical_count = len(clips)
    profile = {
        "source_format": "indexed-videos-nested-jsonl-v1",
        "canonical_format": "indexed-video-segments-v2",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "video_count": len(video_ids),
        "raw_clip_count": raw_clip_count,
        # ``clip_count`` remains the public count used by downstream reports,
        # but now correctly represents searchable canonical segments.
        "clip_count": canonical_count,
        "canonical_clip_count": canonical_count,
        "valid_source_clip_count": raw_clip_count - len(rejected),
        "rejected_clip_count": len(rejected),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "duplicate_segment_group_count": len(duplicate_groups),
        "merged_duplicate_row_count": sum(len(group) - 1 for group in duplicate_groups),
        "exact_duplicate_segment_group_count": len(exact_duplicate_groups),
        "metadata_variant_segment_group_count": (
            len(duplicate_groups) - len(exact_duplicate_groups)
        ),
        "alias_segment_group_count": len(alias_groups),
        "alias_count": sum(
            len(_ordered_unique(item.clip.name for item in group)) - 1
            for group in alias_groups
        ),
        "declared_clip_count_mismatch_count": declared_count_mismatches,
        "duplicate_video_id_count": duplicate_video_id_count,
        "category_counts": dict(sorted(category_counts.items())),
        "raw_coverage": {
            key: raw_coverage[key] / raw_clip_count
            for key in ("clip_description", "clip_goal", "clip_tools", "clip_supplies")
        },
        "coverage": {
            key: coverage_counts[key] / canonical_count
            for key in ("clip_description", "clip_goal", "clip_tools", "clip_supplies")
        },
        "month1_month2_ready": True,
        "structured_search_ready": True,
        "aligned_clip_benchmark_ready": True,
        "original_result_sets_present": False,
        "human_relevance_labels_present": False,
        "comparative_evaluation_ready": False,
        "month3_ready": False,
        "month3_blockers": [
            "No separate human preference labels are included for comparative accuracy.",
            "No precomputed dense embedding vectors are included for a dense baseline.",
            "These items do not block structured top-k search.",
        ],
    }
    return _CanonicalizationResult(clips=clips, rejected=rejected, profile=profile)


def flatten_indexed_videos(source_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return canonical valid segments and a transparent source-data profile."""
    result = _canonicalize_indexed_videos(source_path)
    return result.clips, result.profile


def prepare_indexed_videos(source_path: Path, output_dir: Path) -> dict[str, Path]:
    """Write canonical clips, rejected source rows, and the audit profile."""
    result = _canonicalize_indexed_videos(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_path = output_dir / "indexed_video_clips.jsonl"
    rejected_path = output_dir / "rejected_clips.jsonl"
    profile_path = output_dir / "indexed_video_profile.json"
    write_jsonl(clips_path, result.clips)
    write_jsonl(rejected_path, result.rejected)
    profile_path.write_text(
        json.dumps(result.profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"clips": clips_path, "rejected": rejected_path, "profile": profile_path}
