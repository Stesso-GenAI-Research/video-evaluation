"""Direct-title and exact-phrase controlled benchmark for the IndexedVideo sample.

The nested clip name is treated as a short step query.  Its aligned clip is
the known target, but the candidate representation deliberately excludes that
name.  Lexical, structured, and hybrid retrieval therefore see the same
description, goal, tool, and supply evidence without receiving the answer as
an exact title match.

This is weak supervision from field alignment, not a substitute for human
relevance judgments.  It is useful as a deterministic development benchmark.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from action_semantics.extraction.triples import extract_triples
from action_semantics.io_utils import read_clips, sha256_file, write_csv
from action_semantics.models import ActionTriple, ClipRecord, TextSegment
from action_semantics.retrieval.scorers import (
    STRUCTURED_SCORER_VERSION,
    StructuredResources,
    resources_from_files,
    structured_score,
)
from action_semantics.retrieval.search import imperative_fallback_text
from action_semantics.text import normalize_term


BENCHMARK_NAME = "aligned_clip_leave_name_out_v2"
BOOTSTRAP_SEED = 1729
BOOTSTRAP_ITERATIONS = 2000
METHODS = ("lexical_tfidf", "structured_action", "hybrid")
PRIMARY_METRICS = ("hit_at_1", "hit_at_3", "hit_at_10", "mean_reciprocal_rank")


def _metadata_inventory(clip: ClipRecord, key: str) -> list[str]:
    metadata = clip.gemini_metadata.get("clip", {})
    if not isinstance(metadata, dict):
        return []
    values = metadata.get(key, [])
    output = [
        value.strip()
        for value in values if isinstance(value, str) and value.strip()
    ] if isinstance(values, list) else []
    item_key = "tool_items" if key == "tools" else "supply_items"
    items = metadata.get(item_key, [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("name"), str):
                output.append(item["name"])
            alternatives = item.get("alternatives", [])
            if isinstance(alternatives, list):
                output.extend(value for value in alternatives if isinstance(value, str))
    return list(dict.fromkeys(output))


def _candidate_text(clip: ClipRecord) -> str:
    """Return exactly the candidate fields shared by every benchmark method."""
    values = [
        clip.description,
        clip.summary,
        *_metadata_inventory(clip, "tools"),
        *_metadata_inventory(clip, "supplies"),
    ]
    return " ".join(
        value.strip() for value in values if isinstance(value, str) and value.strip()
    )


def _has_narrative_candidate_text(clip: ClipRecord) -> bool:
    """Candidates require a description or goal, independent of query parsing."""
    return bool((clip.description or "").strip() or (clip.summary or "").strip())


def _normalized_phrase_occurs(query: str | None, candidate: str) -> bool:
    normalized_query = normalize_term(query)
    normalized_candidate = normalize_term(candidate)
    if not normalized_query or not normalized_candidate:
        return False
    # Padding enforces token boundaries, so ``mix`` does not leak through
    # merely because the candidate contains ``mixing``.
    return f" {normalized_query} " in f" {normalized_candidate} "


def _category_name(clip: ClipRecord) -> str:
    source_video = clip.gemini_metadata.get("source_video", {})
    if not isinstance(source_video, dict):
        return "Unknown"
    category = source_video.get("category")
    if isinstance(category, dict):
        value = category.get("name")
    else:
        value = category
    return str(value).strip() if value not in (None, "") else "Unknown"


def _video_cluster_id(clip: ClipRecord) -> str:
    return str(clip.video_id) if clip.video_id not in (None, "") else f"clip:{clip.clip_id}"


def _select_evaluation_queries(
    candidates: list[ClipRecord],
    query_triples_by_id: dict[str, list[ActionTriple]],
) -> tuple[list[ClipRecord], dict[str, Any]]:
    """Apply query-only exclusions without changing the candidate corpus."""
    normalized_titles = [normalize_term(clip.title) for clip in candidates]
    title_counts = Counter(value for value in normalized_titles if value)
    ambiguous: set[str] = set()
    leakage: set[str] = set()
    no_action: set[str] = set()

    for clip, normalized_title in zip(candidates, normalized_titles, strict=True):
        if not normalized_title or title_counts[normalized_title] > 1:
            ambiguous.add(clip.clip_id)
        if _normalized_phrase_occurs(clip.title, _candidate_text(clip)):
            leakage.add(clip.clip_id)
        if not query_triples_by_id.get(clip.clip_id):
            no_action.add(clip.clip_id)

    excluded = ambiguous | leakage | no_action
    eligible = [clip for clip in candidates if clip.clip_id not in excluded]
    parsed_count = len(candidates) - len(no_action)
    report = {
        "candidate_count": len(candidates),
        "parseable_action_query_count": parsed_count,
        "parser_query_coverage": parsed_count / len(candidates) if candidates else 0.0,
        "eligible_query_count": len(eligible),
        "excluded_query_count": len(excluded),
        # These counts intentionally describe each condition independently;
        # a malformed query can satisfy more than one condition.
        "exclusion_counts": {
            "ambiguous_normalized_query_title": len(ambiguous),
            "exact_normalized_query_in_paired_candidate": len(leakage),
            "no_parsed_action": len(no_action),
        },
        "exclusion_counts_may_overlap": True,
    }
    return eligible, report


@dataclass(frozen=True)
class RankOutcome:
    """The target's positive-score rank interval.

    A score of zero means that a method found no evidence for the target, so
    it receives no rank.  Positive score ties are represented as an interval;
    aggregate metrics average over every possible position in that interval.
    ``deterministic_rank`` and ``top_clip_id`` exist only to make row-level
    debugging reproducible.
    """

    deterministic_rank: int | None
    best_rank: int | None
    worst_rank: int | None
    top_clip_id: str | None
    relevant_score: float
    positive_candidate_count: int
    tie_size: int


def _rank(scores: dict[str, float], relevant_id: str) -> RankOutcome:
    if relevant_id not in scores:
        raise ValueError(f"Relevant clip is absent from the candidate scores: {relevant_id}")
    if not scores:
        raise ValueError("Cannot rank an empty candidate set.")
    positive_ids = [clip_id for clip_id, score in scores.items() if score > 0.0]
    ordered = sorted(positive_ids, key=lambda clip_id: (-scores[clip_id], clip_id))
    top_clip_id = ordered[0] if ordered else None
    relevant_score = scores[relevant_id]
    if relevant_score <= 0.0:
        return RankOutcome(
            deterministic_rank=None,
            best_rank=None,
            worst_rank=None,
            top_clip_id=top_clip_id,
            relevant_score=relevant_score,
            positive_candidate_count=len(positive_ids),
            tie_size=0,
        )

    tied_ids = [
        clip_id
        for clip_id in positive_ids
        if isclose(scores[clip_id], relevant_score, rel_tol=1e-12, abs_tol=1e-12)
    ]
    greater_count = sum(
        scores[clip_id] > relevant_score
        and not isclose(scores[clip_id], relevant_score, rel_tol=1e-12, abs_tol=1e-12)
        for clip_id in positive_ids
    )
    return RankOutcome(
        deterministic_rank=ordered.index(relevant_id) + 1,
        best_rank=greater_count + 1,
        worst_rank=greater_count + len(tied_ids),
        top_clip_id=top_clip_id,
        relevant_score=relevant_score,
        positive_candidate_count=len(positive_ids),
        tie_size=len(tied_ids),
    )


def _expected_hit(outcome: RankOutcome, cutoff: int) -> float:
    if outcome.best_rank is None or outcome.worst_rank is None:
        return 0.0
    positions_inside_cutoff = max(
        0, min(cutoff, outcome.worst_rank) - outcome.best_rank + 1
    )
    return positions_inside_cutoff / outcome.tie_size


def _expected_reciprocal_rank(outcome: RankOutcome) -> float:
    if outcome.best_rank is None or outcome.worst_rank is None:
        return 0.0
    return sum(
        1.0 / rank for rank in range(outcome.best_rank, outcome.worst_rank + 1)
    ) / outcome.tie_size


def _metrics(outcomes: list[RankOutcome]) -> dict[str, Any]:
    if not outcomes:
        return {
            "query_count": 0,
            "target_positive_score_count": 0,
            "target_positive_score_rate": None,
            "positive_target_tie_count": 0,
            "positive_target_tie_rate": None,
            "hit_at_1": None,
            "hit_at_3": None,
            "hit_at_10": None,
            "mean_reciprocal_rank": None,
            "median_expected_rank_among_positive_targets": None,
        }
    positive = [outcome for outcome in outcomes if outcome.best_rank is not None]
    tied = [outcome for outcome in positive if outcome.tie_size > 1]
    expected_ranks = [
        (outcome.best_rank + outcome.worst_rank) / 2
        for outcome in positive
        if outcome.best_rank is not None and outcome.worst_rank is not None
    ]
    return {
        "query_count": len(outcomes),
        "target_positive_score_count": len(positive),
        "target_positive_score_rate": len(positive) / len(outcomes),
        "positive_target_tie_count": len(tied),
        "positive_target_tie_rate": len(tied) / len(outcomes),
        "hit_at_1": float(np.mean([_expected_hit(row, 1) for row in outcomes])),
        "hit_at_3": float(np.mean([_expected_hit(row, 3) for row in outcomes])),
        "hit_at_10": float(np.mean([_expected_hit(row, 10) for row in outcomes])),
        "mean_reciprocal_rank": float(
            np.mean([_expected_reciprocal_rank(row) for row in outcomes])
        ),
        "median_expected_rank_among_positive_targets": (
            float(median(expected_ranks)) if expected_ranks else None
        ),
    }


def _metric_values(outcomes: list[RankOutcome]) -> dict[str, float]:
    metrics = _metrics(outcomes)
    return {name: float(metrics[name]) for name in PRIMARY_METRICS}


def _paired_cluster_bootstrap_delta_cis(
    *,
    video_ids: list[str],
    baseline_ranks: list[RankOutcome],
    challenger_ranks: list[RankOutcome],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap paired metric deltas while resampling whole source videos."""
    if not (len(video_ids) == len(baseline_ranks) == len(challenger_ranks)):
        raise ValueError("Video IDs and paired rank lists must have equal lengths.")
    if not video_ids:
        return {
            "cluster_unit": "video",
            "cluster_count": 0,
            "iterations": iterations,
            "seed": seed,
            "metrics": {},
        }
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be at least 1.")

    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, video_id in enumerate(video_ids):
        grouped[video_id].append(row_index)
    cluster_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in PRIMARY_METRICS}

    baseline_point = _metric_values(baseline_ranks)
    challenger_point = _metric_values(challenger_ranks)
    for _ in range(iterations):
        sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_indices = [index for cluster in sampled_clusters for index in grouped[cluster]]
        sampled_baseline = [baseline_ranks[index] for index in sampled_indices]
        sampled_challenger = [challenger_ranks[index] for index in sampled_indices]
        baseline_metrics = _metric_values(sampled_baseline)
        challenger_metrics = _metric_values(sampled_challenger)
        for metric in PRIMARY_METRICS:
            samples[metric].append(challenger_metrics[metric] - baseline_metrics[metric])

    return {
        "cluster_unit": "video",
        "cluster_count": len(cluster_ids),
        "iterations": iterations,
        "seed": seed,
        "metrics": {
            metric: {
                "estimate": challenger_point[metric] - baseline_point[metric],
                "ci_95_low": float(np.percentile(samples[metric], 2.5)),
                "ci_95_high": float(np.percentile(samples[metric], 97.5)),
            }
            for metric in PRIMARY_METRICS
        },
    }


def _task_summary(
    *,
    ranks: dict[str, list[RankOutcome]],
    video_ids: list[str],
) -> dict[str, Any]:
    return {
        "query_count": len(video_ids),
        "score_and_tie_policy": {
            "zero_score": "no retrieved rank; contributes zero to hits and reciprocal rank",
            "positive_score_ties": (
                "metrics use the expected value under a uniform ordering of tied clips"
            ),
            "row_level_display_order": "descending score, then clip_id",
        },
        "methods": {method: _metrics(ranks[method]) for method in METHODS},
        "paired_cluster_bootstrap_95": {
            "structured_action_minus_lexical_tfidf": _paired_cluster_bootstrap_delta_cis(
                video_ids=video_ids,
                baseline_ranks=ranks["lexical_tfidf"],
                challenger_ranks=ranks["structured_action"],
            ),
            "hybrid_minus_lexical_tfidf": _paired_cluster_bootstrap_delta_cis(
                video_ids=video_ids,
                baseline_ranks=ranks["lexical_tfidf"],
                challenger_ranks=ranks["hybrid"],
            ),
        },
    }


def run_field_heldout_benchmark(
    *,
    clips_jsonl: Path,
    month1_dir: Path,
    month2_dir: Path,
    output_dir: Path,
    spacy_model: str,
    hybrid_alpha: float = 0.5,
) -> dict[str, Path]:
    """Compare lexical, structured, and hybrid retrieval on aligned fields."""
    if not 0.0 <= hybrid_alpha <= 1.0:
        raise ValueError("hybrid_alpha must be between 0 and 1.")

    # This fixed corpus is defined only by candidate-side availability.  A
    # query parsing failure never removes a distractor from the search space.
    candidates = [clip for clip in read_clips(clips_jsonl) if _has_narrative_candidate_text(clip)]
    if len(candidates) < 2:
        raise ValueError("The benchmark requires at least two clips with descriptions or goals.")
    candidate_ids = [clip.clip_id for clip in candidates]
    candidate_id_set = set(candidate_ids)

    base = resources_from_files(month1_dir, month2_dir)
    candidate_triples = [
        row
        for row in base.triples
        if row.record_type == "clip"
        and row.record_id in candidate_id_set
        and row.source_field in {"description", "summary"}
    ]

    query_segments = [
        TextSegment(
            record_type="step",
            record_id=clip.clip_id,
            source_field="query",
            text=clip.title or "",
        )
        for clip in candidates
    ]
    query_triples = extract_triples(query_segments, spacy_model)
    query_triples_by_id: dict[str, list[ActionTriple]] = defaultdict(list)
    for row in query_triples:
        query_triples_by_id[row.record_id].append(row)
    initially_parsed_ids = set(query_triples_by_id)
    # The fallback vocabulary is derived from candidate-side narrative text,
    # not from held-out clip titles. VerbNet merely confirms that the first
    # word can function as a verb. Requiring support in another candidate
    # prevents a target's own paired description from deciding its eligibility.
    candidate_action_support: dict[str, set[str]] = defaultdict(set)
    for row in candidate_triples:
        candidate_action_support[row.action_lemma].add(row.record_id)
    known_verbs = {
        row.action_lemma
        for row in base.verbnet
        if row.has_mapping and row.action_lemma in candidate_action_support
    }
    fallback_segments = [
        TextSegment(
            record_type="step",
            record_id=clip.clip_id,
            source_field="query",
            text=rewritten,
        )
        for clip in candidates
        if clip.clip_id not in query_triples_by_id
        if (rewritten := imperative_fallback_text(clip.title or "", known_verbs))
        if candidate_action_support[rewritten.split()[0]] - {clip.clip_id}
    ]
    for row in extract_triples(fallback_segments, spacy_model):
        query_triples_by_id[row.record_id].append(row)
    queries, eligibility = _select_evaluation_queries(candidates, query_triples_by_id)
    fallback_parsed_ids = set(query_triples_by_id) - initially_parsed_ids
    eligibility["imperative_fallback_query_count"] = len(fallback_parsed_ids)
    eligibility["eligible_imperative_fallback_query_count"] = len(
        fallback_parsed_ids & {clip.clip_id for clip in queries}
    )
    if not queries:
        raise ValueError(
            "No direct-leakage-controlled candidate also had a parseable title action."
        )

    documents = [_candidate_text(clip) for clip in candidates]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    candidate_matrix = vectorizer.fit_transform(documents)
    query_matrix = vectorizer.transform([clip.title or "" for clip in queries])
    lexical_matrix = (query_matrix @ candidate_matrix.T).toarray()

    resources = StructuredResources(
        triples=[
            *candidate_triples,
            *(row for clip in queries for row in query_triples_by_id[clip.clip_id]),
        ],
        verbnet=base.verbnet,
        framenet=base.framenet,
        # Taxonomy is retained so its diagnostic field can be reported by the
        # scorer, but it is not part of the structured total score.
        taxonomy=base.taxonomy,
    )
    candidates_by_video: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_video[_video_cluster_id(candidate)].append(candidate.clip_id)

    rows: list[dict[str, Any]] = []
    global_ranks = {method: [] for method in METHODS}
    within_ranks = {method: [] for method in METHODS}
    global_video_ids: list[str] = []
    within_video_ids: list[str] = []
    category_ranks: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {method: [] for method in METHODS}
    )

    for query_index, clip in enumerate(queries):
        lexical = {
            candidate_id: float(lexical_matrix[query_index, candidate_index])
            for candidate_index, candidate_id in enumerate(candidate_ids)
        }
        structured = {
            candidate_id: structured_score(clip.clip_id, candidate_id, resources)[
                "structured_score"
            ]
            for candidate_id in candidate_ids
        }
        hybrid = {
            candidate_id: hybrid_alpha * lexical[candidate_id]
            + (1.0 - hybrid_alpha) * structured[candidate_id]
            for candidate_id in candidate_ids
        }
        scores_by_method = {
            "lexical_tfidf": lexical,
            "structured_action": structured,
            "hybrid": hybrid,
        }
        video_id = _video_cluster_id(clip)
        category = _category_name(clip)
        within_ids = candidates_by_video[video_id]
        result_row: dict[str, Any] = {
            "query_clip_id": clip.clip_id,
            "query_text": clip.title,
            "query_actions": ";".join(
                sorted({row.action_lemma for row in query_triples_by_id[clip.clip_id]})
            ),
            "relevant_clip_id": clip.clip_id,
            "video_id": clip.video_id,
            "category": category,
            "global_candidate_count": len(candidate_ids),
            "within_video_candidate_count": len(within_ids),
        }
        global_video_ids.append(video_id)

        for method, scores in scores_by_method.items():
            outcome = _rank(scores, clip.clip_id)
            global_ranks[method].append(outcome)
            category_ranks[category][method].append(outcome)
            result_row[f"{method}_deterministic_rank"] = outcome.deterministic_rank
            result_row[f"{method}_best_tied_rank"] = outcome.best_rank
            result_row[f"{method}_worst_tied_rank"] = outcome.worst_rank
            result_row[f"{method}_tie_size"] = outcome.tie_size
            result_row[f"{method}_target_has_positive_score"] = (
                outcome.best_rank is not None
            )
            result_row[f"{method}_positive_candidate_count"] = (
                outcome.positive_candidate_count
            )
            result_row[f"{method}_top_clip_id"] = outcome.top_clip_id
            result_row[f"{method}_relevant_score"] = outcome.relevant_score

            if len(within_ids) >= 2:
                within_scores = {candidate_id: scores[candidate_id] for candidate_id in within_ids}
                within_outcome = _rank(within_scores, clip.clip_id)
                within_ranks[method].append(within_outcome)
                result_row[f"within_video_{method}_deterministic_rank"] = (
                    within_outcome.deterministic_rank
                )
                result_row[f"within_video_{method}_best_tied_rank"] = (
                    within_outcome.best_rank
                )
                result_row[f"within_video_{method}_worst_tied_rank"] = (
                    within_outcome.worst_rank
                )
                result_row[f"within_video_{method}_tie_size"] = (
                    within_outcome.tie_size
                )
                result_row[f"within_video_{method}_target_has_positive_score"] = (
                    within_outcome.best_rank is not None
                )
                result_row[f"within_video_{method}_positive_candidate_count"] = (
                    within_outcome.positive_candidate_count
                )
                result_row[f"within_video_{method}_top_clip_id"] = (
                    within_outcome.top_clip_id
                )
            else:
                result_row[f"within_video_{method}_deterministic_rank"] = None
                result_row[f"within_video_{method}_best_tied_rank"] = None
                result_row[f"within_video_{method}_worst_tied_rank"] = None
                result_row[f"within_video_{method}_tie_size"] = None
                result_row[f"within_video_{method}_target_has_positive_score"] = None
                result_row[f"within_video_{method}_positive_candidate_count"] = None
                result_row[f"within_video_{method}_top_clip_id"] = None
        if len(within_ids) >= 2:
            within_video_ids.append(video_id)
        rows.append(result_row)

    global_summary = _task_summary(ranks=global_ranks, video_ids=global_video_ids)
    global_summary["candidate_count"] = len(candidate_ids)
    global_summary["metrics_by_category"] = {
        category: {
            "query_count": len(method_ranks["lexical_tfidf"]),
            "methods": {
                method: _metrics(method_ranks[method]) for method in METHODS
            },
        }
        for category, method_ranks in sorted(category_ranks.items())
    }
    within_summary = _task_summary(ranks=within_ranks, video_ids=within_video_ids)
    within_summary.update(
        {
            "candidate_scope": "clips from the query's source video",
            "minimum_candidate_count": 2,
        }
    )

    summary = {
        "schema_version": "benchmark.v3",
        "benchmark": BENCHMARK_NAME,
        "ground_truth": "weak field alignment: query clip name -> same timestamped clip",
        "query_field": "clip.name/title",
        "candidate_fields": [
            "clip.description",
            "clip.goal/summary",
            "clip.tools",
            "clip.supplies",
        ],
        "excluded_candidate_fields": [
            "clip.name/title",
            "all parent video text and metadata",
        ],
        "candidate_title_excluded": True,
        "corpus_count": len(candidates),
        "query_count": len(queries),
        "query_eligibility": eligibility,
        "hybrid_alpha_lexical": hybrid_alpha,
        "hybrid_alpha_policy": (
            "user-specified fixed value for this run; comparing alphas on this same "
            "benchmark is exploratory and must not be reported as a held-out result"
        ),
        "bootstrap": {
            "cluster_unit": "video",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
        },
        "reproducibility": {
            "clips_sha256": sha256_file(clips_jsonl),
            "triples_sha256": sha256_file(
                month1_dir / "action_object_tool_triples.jsonl"
            ),
            "verbnet_sha256": sha256_file(month1_dir / "verbnet_mappings.jsonl"),
            "framenet_sha256": sha256_file(month2_dir / "framenet_mappings.jsonl"),
            "taxonomy_diagnostic_sha256": sha256_file(
                month2_dir / "diy_actionnet_v1.jsonl"
            ),
            "spacy_model": spacy_model,
            "structured_scorer": STRUCTURED_SCORER_VERSION,
            "taxonomy_used_for_ranking": False,
        },
        "tasks": {
            "global": global_summary,
            "within_video": within_summary,
        },
        # Compatibility aliases for existing report consumers.  These are the
        # global-corpus metrics.
        "methods": global_summary["methods"],
        "limitations": [
            "The known target comes from field alignment, not a separate human judgment.",
            "Results describe narrative candidates after direct-title and exact-phrase controls.",
            "Clip names and descriptions were jointly generated and can still share partial wording or paraphrases.",
            "Descriptions, goals, tools, and supplies may contain model-generated annotation noise.",
            "The exploratory taxonomy is diagnostic-only and does not affect the structured score.",
            "A human-labeled comparison is still required to decide whether new matches are better.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.json"
    queries_path = output_dir / "benchmark_queries.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(queries_path, rows)
    return {"summary": summary_path, "queries": queries_path}
