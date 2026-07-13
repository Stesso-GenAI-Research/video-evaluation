"""Batch comparison artifacts for supplied and action-semantic rankings.

The supplied ranking is treated as data: its order is validated and preserved.
This module deliberately reports set agreement rather than declaring either
ranking better.  Result quality is measured later with the blinded worksheet.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from action_semantics.io_utils import read_clips, write_csv, write_jsonl
from action_semantics.retrieval.evaluation import bootstrap_ci
from action_semantics.retrieval.provenance import build_retrieval_provenance
from action_semantics.retrieval.search import rank_indexed_clips


BLIND_REVIEW_SEED = 1729
ChallengerMethod = Literal["structured", "hybrid"]
_YES_LABELS = {"yes", "y", "true", "1"}
_NO_LABELS = {"no", "n", "false", "0"}
_REVIEW_DIMENSIONS = ("overall_relevant", "action", "object", "tool")
_BOOTSTRAP_DRAWS = 5000


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
    wrong_original_counts = {
        row.step_id: len(row.original_matches)
        for row in inputs
        if len(row.original_matches) != top_k
    }
    if wrong_original_counts:
        raise ValueError(
            f"Every original ranking must contain exactly top_k={top_k} results; "
            f"found: {wrong_original_counts}"
        )
    provenance = build_retrieval_provenance(
        clips_jsonl=clips_jsonl,
        month1_dir=month1_dir,
        month2_dir=month2_dir,
        spacy_model=spacy_model,
    )
    configuration = {
        "challenger_method": challenger_method,
        "requested_top_k": top_k,
        "hybrid_alpha_lexical": (
            hybrid_alpha if challenger_method == "hybrid" else None
        ),
        "spacy_model": spacy_model,
    }

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
                "configuration": configuration,
                "provenance": provenance,
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
        "configuration": configuration,
        "provenance": provenance,
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


def _ranking_ids(rows: Any, *, context: str, allow_empty: bool) -> list[str]:
    if not isinstance(rows, list) or (not rows and not allow_empty):
        expectation = "a list" if allow_empty else "a nonempty list"
        raise ValueError(f"{context} must be {expectation}")
    clip_ids: list[str] = []
    ranks: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{context} contains a non-object result")
        clip_id = row.get("clip_id")
        rank = row.get("rank")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError(f"{context} contains a blank or non-string clip_id")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError(f"{context} contains a non-positive integer rank")
        clip_ids.append(clip_id.strip())
        ranks.append(rank)
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError(f"{context} contains duplicate clip IDs")
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError(f"{context} must be in contiguous rank order starting at 1")
    return clip_ids


def _read_ranking_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Ranking line {line_number} of {path} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"Rankings file is empty: {path}")
    step_ids = [row.get("step_id") for row in rows]
    if any(not isinstance(step_id, str) or not step_id.strip() for step_id in step_ids):
        raise ValueError("Every ranking row must have a nonblank string step_id")
    duplicates = sorted(
        step_id for step_id, count in Counter(step_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Ranking step_id values must be unique; duplicates: {duplicates}")
    return rows


def _parse_rank(value: str | None, *, column: str, review_id: str) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        rank = int(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {column} value {text!r} for review_id {review_id}"
        ) from exc
    if rank < 1 or str(rank) != text:
        raise ValueError(f"Invalid {column} value {text!r} for review_id {review_id}")
    return rank


def _parse_label(value: str | None, *, dimension: str, review_id: str) -> bool | None:
    label = (value or "").strip().lower()
    if not label:
        return None
    if label in _YES_LABELS:
        return True
    if label in _NO_LABELS:
        return False
    raise ValueError(
        f"Invalid {dimension} label {label!r} for review_id {review_id}; "
        "use yes, no, or leave it blank."
    )


def _win_tie_loss(deltas: list[float]) -> dict[str, int]:
    tolerance = 1e-12
    return {
        "challenger_wins": sum(delta > tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "challenger_losses": sum(delta < -tolerance for delta in deltas),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_batch_review(
    rankings_jsonl: Path,
    review_csv: Path,
    output_json: Path,
    seed: int = BLIND_REVIEW_SEED,
) -> dict[str, Any]:
    """Score a completed blinded worksheet against its hidden ranking key.

    A dimension is evaluated for a step only when every pooled candidate for
    that step has a valid label.  Blank cells remain missing; they are never
    converted into negative judgments.
    """
    rankings = _read_ranking_rows(rankings_jsonl)
    assignment_seeds = {
        row.get("blind_review_assignment", {}).get("seed")
        for row in rankings
        if isinstance(row.get("blind_review_assignment"), dict)
    }
    if len(assignment_seeds) != 1 or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in assignment_seeds
    ):
        raise ValueError("Ranking rows must use one valid blind-review seed")
    blind_review_seed = next(iter(assignment_seeds))
    rng = random.Random(blind_review_seed)
    expected_reviews: dict[str, dict[str, Any]] = {}
    step_rankings: dict[str, dict[str, Any]] = {}
    requested_top_k_values: set[int] = set()

    for row in rankings:
        step_id = str(row["step_id"]).strip()
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Ranking step {step_id} has a blank or non-string query")
        original_ids = _ranking_ids(
            row.get("original_matches"),
            context=f"ranking step {step_id} original_matches",
            allow_empty=False,
        )
        challenger = row.get("challenger")
        if not isinstance(challenger, dict):
            raise ValueError(f"Ranking step {step_id} has no challenger object")
        challenger_ids = _ranking_ids(
            challenger.get("matches"),
            context=f"ranking step {step_id} challenger matches",
            allow_empty=True,
        )
        top_k = challenger.get("requested_top_k")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"Ranking step {step_id} has an invalid requested_top_k")
        requested_top_k_values.add(top_k)

        assignment = row.get("blind_review_assignment")
        if not isinstance(assignment, dict):
            raise ValueError(f"Ranking step {step_id} has no blind_review_assignment")
        original_label = assignment.get("original_set")
        challenger_label = assignment.get("challenger_set")
        if {original_label, challenger_label} != {"A", "B"}:
            raise ValueError(
                f"Ranking step {step_id} must assign opposite A/B review labels"
            )
        expected_original_is_a = bool(rng.getrandbits(1))
        expected_original_label = "A" if expected_original_is_a else "B"
        if original_label != expected_original_label:
            raise ValueError(
                f"Ranking step {step_id} blind A/B assignment is inconsistent with seed "
                f"{blind_review_seed}"
            )

        original_rank = {
            clip_id: rank for rank, clip_id in enumerate(original_ids, start=1)
        }
        challenger_rank = {
            clip_id: rank for rank, clip_id in enumerate(challenger_ids, start=1)
        }
        pooled_ids = list(dict.fromkeys([*original_ids, *challenger_ids]))
        rng.shuffle(pooled_ids)
        for candidate_order, clip_id in enumerate(pooled_ids, start=1):
            review_id = f"{step_id}:{candidate_order:03d}"
            rank_a = (
                original_rank.get(clip_id)
                if expected_original_is_a
                else challenger_rank.get(clip_id)
            )
            rank_b = (
                challenger_rank.get(clip_id)
                if expected_original_is_a
                else original_rank.get(clip_id)
            )
            expected_reviews[review_id] = {
                "step_id": step_id,
                "query": query,
                "clip_id": clip_id,
                "candidate_order": candidate_order,
                "set_a_rank": rank_a,
                "set_b_rank": rank_b,
            }
        step_rankings[step_id] = {
            "query": query,
            "top_k": top_k,
            "original_ids": original_ids,
            "challenger_ids": challenger_ids,
            "pooled_ids": pooled_ids,
        }

    if len(requested_top_k_values) != 1:
        raise ValueError(
            "All ranking rows must use the same requested_top_k for aggregate scoring"
        )
    top_k = next(iter(requested_top_k_values))

    with review_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "review_id",
            "step_id",
            "query",
            "candidate_order",
            "set_a_rank",
            "set_b_rank",
            "clip_id",
            *_REVIEW_DIMENSIONS,
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Review CSV is missing required columns: {sorted(missing_columns)}"
            )
        review_rows = list(reader)
    if not review_rows:
        raise ValueError(f"Review CSV has no rows: {review_csv}")

    review_ids = [(row.get("review_id") or "").strip() for row in review_rows]
    duplicate_review_ids = sorted(
        review_id for review_id, count in Counter(review_ids).items() if count > 1
    )
    if duplicate_review_ids:
        raise ValueError(f"Review IDs must be unique; duplicates: {duplicate_review_ids}")
    step_clip_pairs = [
        ((row.get("step_id") or "").strip(), (row.get("clip_id") or "").strip())
        for row in review_rows
    ]
    duplicate_pairs = sorted(
        pair for pair, count in Counter(step_clip_pairs).items() if count > 1
    )
    if duplicate_pairs:
        raise ValueError(f"Review step+clip pairs must be unique; duplicates: {duplicate_pairs}")

    labels: dict[tuple[str, str], dict[str, bool | None]] = {}
    seen_review_ids: set[str] = set()
    for row in review_rows:
        review_id = (row.get("review_id") or "").strip()
        expected = expected_reviews.get(review_id)
        if expected is None:
            raise ValueError(f"Unknown review_id in worksheet: {review_id!r}")
        seen_review_ids.add(review_id)
        step_id = (row.get("step_id") or "").strip()
        clip_id = (row.get("clip_id") or "").strip()
        if step_id != expected["step_id"] or clip_id != expected["clip_id"]:
            raise ValueError(
                f"Review row {review_id} does not match its expected step_id and clip_id"
            )
        if (row.get("query") or "").strip() != expected["query"]:
            raise ValueError(f"Review row {review_id} has a query inconsistent with rankings")
        try:
            candidate_order = int((row.get("candidate_order") or "").strip())
        except ValueError as exc:
            raise ValueError(f"Review row {review_id} has an invalid candidate_order") from exc
        if candidate_order != expected["candidate_order"]:
            raise ValueError(f"Review row {review_id} has an inconsistent candidate_order")
        rank_a = _parse_rank(
            row.get("set_a_rank"), column="set_a_rank", review_id=review_id
        )
        rank_b = _parse_rank(
            row.get("set_b_rank"), column="set_b_rank", review_id=review_id
        )
        if rank_a != expected["set_a_rank"] or rank_b != expected["set_b_rank"]:
            raise ValueError(f"Review row {review_id} has A/B ranks inconsistent with rankings")
        labels[(step_id, clip_id)] = {
            dimension: _parse_label(
                row.get(dimension), dimension=dimension, review_id=review_id
            )
            for dimension in _REVIEW_DIMENSIONS
        }

    missing_review_ids = sorted(set(expected_reviews) - seen_review_ids)
    if missing_review_ids:
        raise ValueError(
            f"Review CSV is missing rows from the blinded worksheet: {missing_review_ids}"
        )

    coverage: dict[str, Any] = {}
    dimension_results: dict[str, Any] = {}
    for dimension in _REVIEW_DIMENSIONS:
        complete_steps: list[str] = []
        partial_steps: list[str] = []
        unlabeled_steps: list[str] = []
        labeled_rows = 0
        for step_id, ranking in step_rankings.items():
            step_labels = [
                labels[(step_id, clip_id)][dimension]
                for clip_id in ranking["pooled_ids"]
            ]
            labeled_rows += sum(value is not None for value in step_labels)
            if all(value is not None for value in step_labels):
                complete_steps.append(step_id)
            elif any(value is not None for value in step_labels):
                partial_steps.append(step_id)
            else:
                unlabeled_steps.append(step_id)

        coverage[dimension] = {
            "labeled_rows": labeled_rows,
            "total_rows": len(expected_reviews),
            "row_coverage": labeled_rows / len(expected_reviews),
            "complete_steps": len(complete_steps),
            "partial_steps": len(partial_steps),
            "unlabeled_steps": len(unlabeled_steps),
            "total_steps": len(step_rankings),
            "complete_step_coverage": len(complete_steps) / len(step_rankings),
        }

        per_step: list[dict[str, Any]] = []
        precision_deltas: list[float] = []
        success_deltas: list[float] = []
        original_precisions: list[float] = []
        challenger_precisions: list[float] = []
        original_successes: list[float] = []
        challenger_successes: list[float] = []
        for step_id in complete_steps:
            ranking = step_rankings[step_id]
            relevance = {
                clip_id: bool(labels[(step_id, clip_id)][dimension])
                for clip_id in ranking["pooled_ids"]
            }
            original_top_k = ranking["original_ids"][:top_k]
            challenger_top_k = ranking["challenger_ids"][:top_k]
            original_precision = sum(relevance[clip_id] for clip_id in original_top_k) / top_k
            challenger_precision = (
                sum(relevance[clip_id] for clip_id in challenger_top_k) / top_k
            )
            original_success = float(any(relevance[clip_id] for clip_id in original_top_k))
            challenger_success = float(
                any(relevance[clip_id] for clip_id in challenger_top_k)
            )
            precision_delta = challenger_precision - original_precision
            success_delta = challenger_success - original_success
            original_precisions.append(original_precision)
            challenger_precisions.append(challenger_precision)
            original_successes.append(original_success)
            challenger_successes.append(challenger_success)
            precision_deltas.append(precision_delta)
            success_deltas.append(success_delta)
            per_step.append(
                {
                    "step_id": step_id,
                    "query": ranking["query"],
                    "original_precision_at_k": original_precision,
                    "challenger_precision_at_k": challenger_precision,
                    "precision_delta_challenger_minus_original": precision_delta,
                    "original_success_at_k": original_success,
                    "challenger_success_at_k": challenger_success,
                    "success_delta_challenger_minus_original": success_delta,
                }
            )

        bootstrap = bootstrap_ci(
            precision_deltas,
            seed=seed,
            draws=_BOOTSTRAP_DRAWS,
            confidence=0.95,
        )
        dimension_results[dimension] = {
            "k": top_k,
            "scored_steps": len(complete_steps),
            "original": {
                "mean_precision_at_k": _mean(original_precisions),
                "success_at_k": _mean(original_successes),
            },
            "challenger": {
                "mean_precision_at_k": _mean(challenger_precisions),
                "success_at_k": _mean(challenger_successes),
            },
            "delta_challenger_minus_original": {
                "mean_precision_at_k": _mean(precision_deltas),
                "success_at_k": _mean(success_deltas),
            },
            "precision_wins_ties_losses": _win_tie_loss(precision_deltas),
            "success_wins_ties_losses": _win_tie_loss(success_deltas),
            "paired_bootstrap_95_ci_mean_precision_delta": {
                "unit": "step",
                "seed": seed,
                "iterations": _BOOTSTRAP_DRAWS,
                "n": bootstrap["n"],
                "estimate": bootstrap["mean"],
                "lower": bootstrap["lower"],
                "upper": bootstrap["upper"],
                "confidence": bootstrap["confidence"],
            },
            "per_step": per_step,
        }

    report = {
        "schema_version": "batch_comparison.review_scores.v1",
        "source_rankings_jsonl": str(rankings_jsonl),
        "source_review_csv": str(review_csv),
        "blind_review_seed": blind_review_seed,
        "bootstrap_seed": seed,
        "k": top_k,
        "judgment_coverage": {
            "review_rows": len(expected_reviews),
            "steps": len(step_rankings),
            "dimensions": coverage,
        },
        "dimensions": dimension_results,
        "note": (
            "A step is scored for a dimension only when every pooled candidate has a "
            "yes/no judgment. Blank labels remain missing and are not counted as no."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
