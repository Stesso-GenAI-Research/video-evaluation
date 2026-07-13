"""Descriptive comparison of two top-k result sets.

This module intentionally does not declare a winner.  A ranking method cannot
establish its own correctness by scoring the results it selected.  Quality
claims require aligned ground truth or blinded human judgments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from action_semantics.io_utils import read_clips
from action_semantics.retrieval.search import rank_indexed_clips


ChallengerMethod = Literal["structured", "hybrid"]


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _rerank(rows: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    by_id = {row["clip_id"]: row for row in rows}
    output: list[dict[str, Any]] = []
    for rank, clip_id in enumerate(ids, start=1):
        row = dict(by_id[clip_id])
        row["rank"] = rank
        output.append(row)
    return output


def compare_result_sets(
    *,
    query_text: str,
    clips_jsonl: Path,
    month1_dir: Path,
    month2_dir: Path,
    spacy_model: str,
    top_k: int = 3,
    original_clip_ids: list[str] | None = None,
    challenger_method: ChallengerMethod = "hybrid",
    hybrid_alpha: float = 0.5,
) -> dict[str, Any]:
    """Diff an explicit old ranking or lexical baseline against a challenger."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    clips = read_clips(clips_jsonl)
    corpus_ids = {clip.clip_id for clip in clips}

    lexical_search = rank_indexed_clips(
        query_text=query_text,
        clips_jsonl=clips_jsonl,
        month1_dir=month1_dir,
        month2_dir=month2_dir,
        spacy_model=spacy_model,
        top_k=len(clips),
        method="lexical",
        include_zero_scores=True,
    )
    challenger_search = rank_indexed_clips(
        query_text=query_text,
        clips_jsonl=clips_jsonl,
        month1_dir=month1_dir,
        month2_dir=month2_dir,
        spacy_model=spacy_model,
        top_k=len(clips),
        method=challenger_method,
        hybrid_alpha=hybrid_alpha,
        include_zero_scores=True,
    )

    if original_clip_ids is not None:
        if not original_clip_ids:
            raise ValueError("original_clip_ids was supplied but is empty.")
        if len(set(original_clip_ids)) != len(original_clip_ids):
            raise ValueError("original_clip_ids contains duplicate IDs.")
        missing = [clip_id for clip_id in original_clip_ids if clip_id not in corpus_ids]
        if missing:
            raise ValueError(f"Original result IDs are not in the indexed corpus: {missing}")
        reference_ids = original_clip_ids[:top_k]
        reference_label = "provided_original"
        reference_source = "explicit_original_clip_ids"
    else:
        reference_ids = [row["clip_id"] for row in lexical_search["results"][:top_k]]
        reference_label = "lexical_baseline"
        reference_source = "generated_tfidf_baseline"

    challenger_ids = [
        row["clip_id"] for row in challenger_search["results"][:top_k]
    ]
    # The challenger rows contain every score decomposition needed to inspect
    # both sets under the same field/scorer policy.
    all_scored_rows = challenger_search["results"]
    reference_rows = _rerank(all_scored_rows, reference_ids)
    challenger_rows = _rerank(all_scored_rows, challenger_ids)
    overlap = [clip_id for clip_id in reference_ids if clip_id in set(challenger_ids)]
    reference_ranks = {clip_id: rank for rank, clip_id in enumerate(reference_ids, start=1)}
    challenger_ranks = {
        clip_id: rank for rank, clip_id in enumerate(challenger_ids, start=1)
    }
    return {
        "schema_version": "comparison.v2",
        "query": query_text,
        "top_k": top_k,
        "reference": {
            "label": reference_label,
            "source": reference_source,
            "results": reference_rows,
        },
        "challenger": {
            "label": "action_semantic_search",
            "method": challenger_method,
            "results": challenger_rows,
        },
        "set_difference": {
            "overlap_clip_ids": overlap,
            "overlap_count": len(overlap),
            "jaccard": _jaccard(reference_ids, challenger_ids),
            "reference_only_clip_ids": [
                clip_id for clip_id in reference_ids if clip_id not in challenger_ranks
            ],
            "challenger_only_clip_ids": [
                clip_id for clip_id in challenger_ids if clip_id not in reference_ranks
            ],
            "shared_rank_changes": {
                clip_id: reference_ranks[clip_id] - challenger_ranks[clip_id]
                for clip_id in overlap
            },
        },
        "quality_claim": False,
        "winner": None,
        "interpretation": (
            "This report shows how the rankings differ. It does not prove which set is "
            "better; that requires the aligned benchmark or blinded human judgments."
        ),
        "warnings": [*lexical_search["warnings"], *challenger_search["warnings"]],
    }


def write_comparison_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
