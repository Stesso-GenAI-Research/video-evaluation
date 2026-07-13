"""Batch comparison artifacts for supplied and action-semantic rankings.

The supplied ranking is treated as data: its order is validated and preserved.
This module deliberately reports set agreement rather than declaring either
ranking better.  Result quality is measured later with the blinded worksheet.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from action_semantics.io_utils import read_clips, write_csv, write_jsonl
from action_semantics.retrieval.search import rank_indexed_clips


BLIND_REVIEW_SEED = 1729
ChallengerMethod = Literal["structured", "hybrid"]


class OriginalMatchInput(BaseModel):
    """One result from the supplied, pre-existing ranking."""

    model_config = ConfigDict(extra="forbid", strict=True)

    clip_id: str
    rank: int = Field(ge=1)

    @field_validator("clip_id")
    @classmethod
    def clip_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("clip_id must not be blank")
        return value.strip()


class StepComparisonInput(BaseModel):
    """A query and the exact result order produced by the original system."""

    model_config = ConfigDict(extra="forbid", strict=True)

    step_id: str
    query: str
    original_matches: list[OriginalMatchInput] = Field(min_length=1)

    @field_validator("step_id", "query")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def originals_are_unique_and_ranked_in_supplied_order(self) -> "StepComparisonInput":
        clip_ids = [row.clip_id for row in self.original_matches]
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("original_matches contains duplicate clip_id values")
        ranks = [row.rank for row in self.original_matches]
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(
                "original_matches must be supplied in contiguous rank order starting at 1"
            )
        return self


def _read_inputs(path: Path) -> list[StepComparisonInput]:
    rows: list[StepComparisonInput] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            try:
                rows.append(StepComparisonInput.model_validate(value))
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid comparison input on line {line_number} of {path}:\n{exc}"
                ) from exc
    if not rows:
        raise ValueError(f"Comparison input is empty: {path}")
    step_ids = [row.step_id for row in rows]
    duplicates = sorted({step_id for step_id in step_ids if step_ids.count(step_id) > 1})
    if duplicates:
        raise ValueError(f"step_id values must be unique; duplicates: {duplicates}")
    return rows


def _clip_fields(clip: Any) -> dict[str, Any]:
    clip_metadata = clip.gemini_metadata.get("clip", {})
    source_video = clip.gemini_metadata.get("source_video", {})
    if not isinstance(clip_metadata, dict):
        clip_metadata = {}
    if not isinstance(source_video, dict):
        source_video = {}
    return {
        "clip_id": clip.clip_id,
        "segment_id": clip_metadata.get("segment_id", clip.clip_id),
        "video_id": clip.video_id,
        "clip_title": clip.title,
        "video_title": source_video.get("title"),
        "url": clip.url,
        "start_seconds": clip_metadata.get("start_seconds"),
        "end_seconds": clip_metadata.get("end_seconds"),
    }


def _ranked_clip(clip: Any, rank: int) -> dict[str, Any]:
    return {"rank": rank, **_clip_fields(clip)}


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def run_batch_comparison(
    *,
    comparisons_jsonl: Path,
    clips_jsonl: Path,
    month1_dir: Path,
    month2_dir: Path,
    output_dir: Path,
    spacy_model: str,
    challenger_method: ChallengerMethod = "hybrid",
    top_k: int = 3,
    hybrid_alpha: float = 0.5,
) -> dict[str, Path]:
    """Compare supplied rankings with structured or hybrid top-k results.

    All input and corpus references are validated before any artifact is
    written.  The returned summary is descriptive only; the blind worksheet is
    the mechanism for making later quality judgments.
    """
    if challenger_method not in {"structured", "hybrid"}:
        raise ValueError("challenger_method must be structured or hybrid")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0.0 <= hybrid_alpha <= 1.0:
        raise ValueError("hybrid_alpha must be between 0 and 1")

    inputs = _read_inputs(comparisons_jsonl)
    clips = read_clips(clips_jsonl)
    clips_by_id = {clip.clip_id: clip for clip in clips}
    if len(clips_by_id) != len(clips):
        raise ValueError("clips_jsonl contains duplicate clip_id values")

    missing_by_step: dict[str, list[str]] = {}
    for row in inputs:
        missing = [
            match.clip_id
            for match in row.original_matches
            if match.clip_id not in clips_by_id
        ]
        if missing:
            missing_by_step[row.step_id] = missing
    if missing_by_step:
        raise ValueError(f"Original result IDs are not in the indexed corpus: {missing_by_step}")

    # Rank every query before writing so a parser or search failure cannot leave
    # a partially completed experiment directory.
    searches: list[dict[str, Any]] = []
    for row in inputs:
        searches.append(
            rank_indexed_clips(
                query_text=row.query,
                clips_jsonl=clips_jsonl,
                month1_dir=month1_dir,
                month2_dir=month2_dir,
                spacy_model=spacy_model,
                top_k=top_k,
                method=challenger_method,
                hybrid_alpha=hybrid_alpha,
            )
        )

    rng = random.Random(BLIND_REVIEW_SEED)
    ranking_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    overlap_counts: list[int] = []
    jaccards: list[float] = []
    challenger_counts: list[int] = []

    for input_row, search in zip(inputs, searches, strict=True):
        original_ids = [row.clip_id for row in input_row.original_matches]
        challenger_ids = [str(row["clip_id"]) for row in search["results"]]
        if len(set(challenger_ids)) != len(challenger_ids):
            raise ValueError(
                f"Search returned duplicate clip IDs for step {input_row.step_id}"
            )
        missing_challenger = [
            clip_id for clip_id in challenger_ids if clip_id not in clips_by_id
        ]
        if missing_challenger:
            raise ValueError(
                f"Search returned unknown clip IDs for step {input_row.step_id}: "
                f"{missing_challenger}"
            )

        original_rank = {
            match.clip_id: match.rank for match in input_row.original_matches
        }
        challenger_rank = {
            clip_id: rank for rank, clip_id in enumerate(challenger_ids, start=1)
        }
        overlap_ids = [clip_id for clip_id in original_ids if clip_id in challenger_rank]
        overlap_count = len(overlap_ids)
        set_jaccard = _jaccard(original_ids, challenger_ids)
        overlap_counts.append(overlap_count)
        jaccards.append(set_jaccard)
        challenger_counts.append(len(challenger_ids))

        original_is_a = bool(rng.getrandbits(1))
        original_label = "A" if original_is_a else "B"
        challenger_label = "B" if original_is_a else "A"
        ranking_rows.append(
            {
                "schema_version": "batch_comparison.rankings.v1",
                "step_id": input_row.step_id,
                "query": input_row.query,
                "original_matches": [
                    _ranked_clip(clips_by_id[match.clip_id], match.rank)
                    for match in input_row.original_matches
                ],
                "challenger": {
                    "method": challenger_method,
                    "requested_top_k": top_k,
                    "returned_count": len(challenger_ids),
                    "matches": [
                        _ranked_clip(clips_by_id[clip_id], rank)
                        for rank, clip_id in enumerate(challenger_ids, start=1)
                    ],
                },
                "set_comparison": {
                    "overlap_clip_ids": overlap_ids,
                    "overlap_count": overlap_count,
                    "jaccard": set_jaccard,
                },
                "blind_review_assignment": {
                    "seed": BLIND_REVIEW_SEED,
                    "original_set": original_label,
                    "challenger_set": challenger_label,
                },
                "quality_claim": False,
                "winner": None,
            }
        )

        pooled_ids = list(dict.fromkeys([*original_ids, *challenger_ids]))
        rng.shuffle(pooled_ids)
        for candidate_order, clip_id in enumerate(pooled_ids, start=1):
            clip_fields = _clip_fields(clips_by_id[clip_id])
            rank_a = original_rank.get(clip_id) if original_is_a else challenger_rank.get(clip_id)
            rank_b = challenger_rank.get(clip_id) if original_is_a else original_rank.get(clip_id)
            review_rows.append(
                {
                    "review_id": f"{input_row.step_id}:{candidate_order:03d}",
                    "step_id": input_row.step_id,
                    "query": input_row.query,
                    "candidate_order": candidate_order,
                    "set_a_rank": rank_a if rank_a is not None else "",
                    "set_b_rank": rank_b if rank_b is not None else "",
                    **clip_fields,
                    "overall_relevant": "",
                    "action": "",
                    "object": "",
                    "tool": "",
                    "notes": "",
                }
            )

    step_count = len(inputs)
    requested_slots = step_count * top_k
    returned_slots = sum(challenger_counts)
    summary = {
        "schema_version": "batch_comparison.summary.v1",
        "challenger_method": challenger_method,
        "requested_top_k": top_k,
        "counts": {
            "steps": step_count,
            "original_results": sum(len(row.original_matches) for row in inputs),
            "challenger_results": returned_slots,
            "blind_review_candidates": len(review_rows),
        },
        "coverage": {
            "steps_with_any_challenger_results": sum(count > 0 for count in challenger_counts),
            "steps_with_full_challenger_top_k": sum(
                count == top_k for count in challenger_counts
            ),
            "challenger_slots_requested": requested_slots,
            "challenger_slots_returned": returned_slots,
            "challenger_slot_coverage": returned_slots / requested_slots,
        },
        "overlap": {
            "steps_with_any_overlap": sum(count > 0 for count in overlap_counts),
            "total_shared_clips": sum(overlap_counts),
            "mean_shared_clips_per_step": sum(overlap_counts) / step_count,
        },
        "jaccard": {
            "mean": sum(jaccards) / step_count,
            "minimum": min(jaccards),
            "maximum": max(jaccards),
        },
        "quality_claim": False,
        "winner": None,
    }

    rankings_path = output_dir / "rankings.jsonl"
    summary_path = output_dir / "comparison_summary.json"
    review_path = output_dir / "blind_review.csv"
    write_jsonl(rankings_path, ranking_rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    review_fieldnames = [
        "review_id",
        "step_id",
        "query",
        "candidate_order",
        "set_a_rank",
        "set_b_rank",
        "clip_id",
        "segment_id",
        "video_id",
        "clip_title",
        "video_title",
        "url",
        "start_seconds",
        "end_seconds",
        "overall_relevant",
        "action",
        "object",
        "tool",
        "notes",
    ]
    write_csv(review_path, review_rows, fieldnames=review_fieldnames)
    return {
        "rankings": rankings_path,
        "summary": summary_path,
        "blind_review": review_path,
    }
