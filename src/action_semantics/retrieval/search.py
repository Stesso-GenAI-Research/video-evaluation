"""Functional lexical, structured, and hybrid search over indexed clips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from action_semantics.extraction.triples import extract_triples
from action_semantics.io_utils import read_clips, sha256_file
from action_semantics.models import TextSegment
from action_semantics.retrieval.lexical import (
    PRODUCTION_CANDIDATE_FIELDS,
    TfidfIndex,
    tfidf_scores,
)
from action_semantics.retrieval.scorers import (
    STRUCTURED_SCORER_VERSION,
    StructuredResources,
    resources_from_files,
    structured_score_for_triples,
)
from action_semantics.text import normalize_term


QUERY_ID = "__search_query__"
SearchMethod = Literal["lexical", "structured", "hybrid"]


def _query_triples(query_text: str, spacy_model: str) -> list[Any]:
    return extract_triples(
        [
            TextSegment(
                record_type="step",
                record_id=QUERY_ID,
                source_field="query",
                text=query_text,
            )
        ],
        spacy_model,
    )


def imperative_fallback_text(query_text: str, known_verbs: set[str]) -> str | None:
    """Rewrite terse ``verb object`` text only when the first word is a known verb."""
    words = normalize_term(query_text).split()
    if len(words) < 2 or words[0] not in known_verbs:
        return None
    return f"{words[0]} the {' '.join(words[1:])}"


def _imperative_fallback_triples(
    query_text: str,
    spacy_model: str,
    known_verbs: set[str],
) -> list[Any]:
    rewritten = imperative_fallback_text(query_text, known_verbs)
    return _query_triples(rewritten, spacy_model) if rewritten else []


def _result_metadata(clip: Any) -> dict[str, Any]:
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
        "clip_aliases": clip_metadata.get("aliases", []),
        "clip_description": clip.description,
        "video_title": source_video.get("title"),
        "url": clip.url,
        "start_seconds": clip_metadata.get("start_seconds"),
        "end_seconds": clip_metadata.get("end_seconds"),
    }


def rank_indexed_clips(
    *,
    query_text: str,
    clips_jsonl: Path,
    month1_dir: Path,
    month2_dir: Path,
    spacy_model: str,
    top_k: int = 3,
    method: SearchMethod = "hybrid",
    hybrid_alpha: float = 0.5,
    max_per_video: int | None = None,
    include_zero_scores: bool = False,
    preloaded_clips: list[Any] | None = None,
    preloaded_tfidf: TfidfIndex | None = None,
    preloaded_resources: StructuredResources | None = None,
    precomputed_clips_sha256: str | None = None,
) -> dict[str, Any]:
    """Rank clips and return a stable, explainable search response.

    ``hybrid_alpha`` is the lexical share of a hybrid score.  Structured-only
    search requires a parseable action.  Lexical and hybrid modes remain usable
    when the action parser misses a short or unusual query.
    """
    query_text = query_text.strip()
    if not query_text:
        raise ValueError("Search query must not be blank.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if method not in {"lexical", "structured", "hybrid"}:
        raise ValueError("method must be lexical, structured, or hybrid.")
    if not 0.0 <= hybrid_alpha <= 1.0:
        raise ValueError("hybrid_alpha must be between 0 and 1.")
    if max_per_video is not None and max_per_video < 1:
        raise ValueError("max_per_video must be at least 1 when provided.")

    clips = preloaded_clips if preloaded_clips is not None else read_clips(clips_jsonl)
    lexical = (
        preloaded_tfidf.scores(query_text)
        if preloaded_tfidf is not None
        else tfidf_scores(query_text, clips)
    )
    query_triples = _query_triples(query_text, spacy_model)
    warnings: list[str] = []
    base: StructuredResources | None = preloaded_resources
    if not query_triples and method in {"structured", "hybrid"}:
        base = resources_from_files(month1_dir, month2_dir)
        query_triples = _imperative_fallback_triples(
            query_text,
            spacy_model,
            {row.action_lemma for row in base.verbnet if row.has_mapping},
        )
        if query_triples:
            warnings.append(
                "The parser treated the terse query as a noun phrase, so search retried "
                "it as an imperative instruction."
            )
    if not query_triples:
        if method == "structured":
            raise ValueError(
                "Structured search could not identify an action. Try an imperative query "
                "such as 'remove the old faucet', or use lexical/hybrid search."
            )
        if method == "hybrid":
            warnings.append(
                "The action parser found no verb in the query. "
                "Lexical ranking was used instead."
            )

    structured: dict[str, dict[str, float]] = {}
    if query_triples and method in {"structured", "hybrid"}:
        base = base or resources_from_files(month1_dir, month2_dir)
        structured = {
            clip.clip_id: structured_score_for_triples(
                query_triples, clip.clip_id, base
            )
            for clip in clips
        }

    zero_signals = {
        "structured_score": 0.0,
        "action_match": 0.0,
        "exact_action_match": 0.0,
        "object_match": 0.0,
        "context_match": 0.0,
        "tool_match": 0.0,
        "supply_match": 0.0,
        "scope_match": 0.0,
        "verbnet_match": 0.0,
        "framenet_match": 0.0,
        "taxonomy_match": 0.0,
    }
    scored: list[dict[str, Any]] = []
    for clip in clips:
        parts = structured.get(clip.clip_id, zero_signals)
        lexical_score = lexical[clip.clip_id]
        if method == "lexical" or not query_triples:
            score = lexical_score
        elif method == "structured":
            score = parts["structured_score"]
        else:
            score = (
                hybrid_alpha * lexical_score
                + (1.0 - hybrid_alpha) * parts["structured_score"]
            )
        scored.append(
            {
                **_result_metadata(clip),
                "score": float(score),
                "signals": {
                    "lexical": float(lexical_score),
                    "structured": parts["structured_score"],
                    "action": parts["action_match"],
                    "exact_action": parts["exact_action_match"],
                    "object": parts["object_match"],
                    "tool_or_supply_context": parts["context_match"],
                    "tool": parts["tool_match"],
                    "supply": parts["supply_match"],
                    "location_or_scope_diagnostic": parts["scope_match"],
                    "verbnet": parts["verbnet_match"],
                    "framenet": parts["framenet_match"],
                    # Diagnostic only. It is not part of structured_score.
                    "taxonomy_diagnostic": parts["taxonomy_match"],
                },
            }
        )
    scored.sort(
        key=lambda row: (
            -row["score"],
            -row["signals"]["action"],
            -row["signals"]["object"],
            str(row["clip_id"]),
        )
    )

    selected: list[dict[str, Any]] = []
    video_counts: dict[str | None, int] = {}
    for row in scored:
        if row["score"] <= 0.0 and not include_zero_scores:
            continue
        video_id = row["video_id"]
        if max_per_video is not None and video_counts.get(video_id, 0) >= max_per_video:
            continue
        row["rank"] = len(selected) + 1
        selected.append(row)
        video_counts[video_id] = video_counts.get(video_id, 0) + 1
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        warnings.append(
            f"Only {len(selected)} candidates had a positive {method} score; "
            "zero-score clips were not returned as arbitrary matches."
        )

    return {
        "schema_version": "search.v3",
        "query": {
            "text": query_text,
            "actions": sorted({row.action_lemma for row in query_triples}),
            "objects": sorted(
                {term for row in query_triples for term in row.object_lemmas}
            ),
            "with_or_using_context": sorted(
                {
                    term
                    for row in query_triples
                    for term in (row.tool_lemmas or row.context_tool_lemmas)
                }
            ),
            "location_or_scope": sorted(
                {
                    term
                    for row in query_triples
                    for term in (
                        row.material_lemmas or row.context_material_lemmas
                    )
                }
            ),
        },
        "requested_method": method,
        "method": (
            "lexical_fallback"
            if method == "hybrid" and not query_triples
            else method
        ),
        "requested_hybrid_alpha_lexical": (
            hybrid_alpha if method == "hybrid" else None
        ),
        "hybrid_alpha_lexical": (
            (1.0 if not query_triples else hybrid_alpha)
            if method == "hybrid"
            else None
        ),
        "requested_top_k": top_k,
        "returned_count": len(selected),
        "max_per_video": max_per_video,
        "index": {
            "clips_sha256": precomputed_clips_sha256 or sha256_file(clips_jsonl),
            "canonical_schema": "indexed-video-segments-v2",
            "canonical_clip_count": len(clips),
            "production_lexical_fields": PRODUCTION_CANDIDATE_FIELDS,
            "structured_scorer": STRUCTURED_SCORER_VERSION,
            "taxonomy_used_for_ranking": False,
        },
        "results": selected,
        "warnings": warnings,
    }


def write_search_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
