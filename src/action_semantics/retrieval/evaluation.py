from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from action_semantics.models import PairwiseEvaluationRow


def predict_pairwise_winner(score_a: float | None, score_b: float | None, clip_a_id: str, clip_b_id: str) -> tuple[str | None, bool]:
    if score_a is None or score_b is None:
        return None, False
    if np.isclose(score_a, score_b, rtol=0.0, atol=1e-12):
        return None, True
    return (clip_a_id if score_a > score_b else clip_b_id), False


def pairwise_accuracy(rows: list[PairwiseEvaluationRow], *, count_ties_as_half: bool = True) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": None, "tie_count": 0}
    values: list[float] = []
    tie_count = 0
    missing_count = 0
    for row in rows:
        if row.correct is None:
            if row.tie:
                tie_count += 1
                values.append(0.5 if count_ties_as_half else 0.0)
            else:
                missing_count += 1
            continue
        values.append(1.0 if row.correct else 0.0)
    return {
        "n": len(rows),
        "scored_n": len(values),
        "tie_count": tie_count,
        "missing_count": missing_count,
        "accuracy": float(np.mean(values)) if values else None,
    }


def bootstrap_ci(values: list[float], *, seed: int = 1729, draws: int = 5000, confidence: float = 0.95) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "lower": None, "upper": None, "confidence": confidence}
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = np.empty(draws, dtype=float)
    for i in range(draws):
        means[i] = np.mean(rng.choice(array, size=array.shape[0], replace=True))
    alpha = 1.0 - confidence
    return {
        "n": int(array.shape[0]),
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(means, alpha / 2.0)),
        "upper": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "confidence": confidence,
    }


def clustered_bootstrap_mean_ci(
    values: list[float],
    cluster_ids: list[str],
    *,
    seed: int = 1729,
    draws: int = 5000,
    confidence: float = 0.95,
    cluster_unit: str = "step_id",
) -> dict[str, Any]:
    """Estimate a row-weighted mean CI by resampling whole clusters.

    Each bootstrap draw samples the sorted cluster IDs with replacement and
    includes every row belonging to each sampled cluster. Cluster sorting makes
    a fixed seed reproducible regardless of the clusters' first-seen order.
    """
    if len(values) != len(cluster_ids):
        raise ValueError("Values and cluster IDs must have equal lengths.")
    if draws < 1:
        raise ValueError("Bootstrap draws must be at least 1.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence must be between 0 and 1.")

    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, cluster_id in enumerate(cluster_ids):
        grouped[cluster_id].append(row_index)
    sorted_cluster_ids = sorted(grouped)

    result: dict[str, Any] = {
        "n": len(values),
        "cluster_count": len(sorted_cluster_ids),
        "mean": None,
        "lower": None,
        "upper": None,
        "confidence": confidence,
        "draws": draws,
        "seed": seed,
        "cluster_unit": cluster_unit,
    }
    if not values:
        return result

    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = np.empty(draws, dtype=float)
    for draw_index in range(draws):
        sampled_clusters = rng.choice(
            sorted_cluster_ids,
            size=len(sorted_cluster_ids),
            replace=True,
        )
        sampled_indices = [
            row_index
            for cluster_id in sampled_clusters
            for row_index in grouped[str(cluster_id)]
        ]
        means[draw_index] = np.mean(array[sampled_indices])

    alpha = 1.0 - confidence
    result.update(
        {
            "mean": float(np.mean(array)),
            "lower": float(np.quantile(means, alpha / 2.0)),
            "upper": float(np.quantile(means, 1.0 - alpha / 2.0)),
        }
    )
    return result


def pairwise_accuracy_with_ci(rows: list[PairwiseEvaluationRow], *, seed: int = 1729) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        if row.correct is None:
            if row.tie:
                values.append(0.5)
            continue
        values.append(1.0 if row.correct else 0.0)
    summary = pairwise_accuracy(rows)
    summary["bootstrap_ci"] = bootstrap_ci(values, seed=seed)
    return summary


def dcg(relevances: list[float], k: int) -> float:
    total = 0.0
    for rank, relevance in enumerate(relevances[:k], start=1):
        total += (2.0**relevance - 1.0) / np.log2(rank + 1)
    return float(total)


def ndcg_at_k(relevances: list[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances, k) / ideal_dcg


def reciprocal_rank(relevances: list[float]) -> float:
    for index, relevance in enumerate(relevances, start=1):
        if relevance > 0:
            return 1.0 / index
    return 0.0


def recall_at_k(relevances: list[float], k: int) -> float:
    total_relevant = sum(1 for relevance in relevances if relevance > 0)
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(1 for relevance in relevances[:k] if relevance > 0)
    return retrieved_relevant / total_relevant


def ranking_metrics_from_dataframe(df: pd.DataFrame, *, k_values: tuple[int, ...] = (5, 10)) -> dict[str, Any]:
    required = {"step_id", "score", "relevance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ranking metrics require columns: {sorted(required)}; missing: {sorted(missing)}")
    grouped = defaultdict(dict)
    for step_id, group in df.sort_values(["step_id", "score"], ascending=[True, False]).groupby("step_id"):
        relevances = [float(value) for value in group["relevance"].tolist()]
        grouped[step_id]["mrr"] = reciprocal_rank(relevances)
        for k in k_values:
            grouped[step_id][f"ndcg_at_{k}"] = ndcg_at_k(relevances, k)
            grouped[step_id][f"recall_at_{k}"] = recall_at_k(relevances, k)
    if not grouped:
        return {"step_count": 0}
    metric_names = sorted(next(iter(grouped.values())).keys())
    return {
        "step_count": len(grouped),
        **{
            metric: float(np.mean([values[metric] for values in grouped.values()]))
            for metric in metric_names
        },
    }


def write_evaluation_summary(path: Path, by_model: dict[str, list[PairwiseEvaluationRow]], *, seed: int = 1729) -> dict[str, Any]:
    summary = {model: pairwise_accuracy_with_ci(rows, seed=seed) for model, rows in by_model.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
