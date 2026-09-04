import pytest

from action_semantics.retrieval.evaluation import (
    clustered_bootstrap_mean_ci,
    ndcg_at_k,
    predict_pairwise_winner,
    reciprocal_rank,
)


def test_predict_pairwise_winner_tie():
    predicted, tie = predict_pairwise_winner(0.5, 0.5, "a", "b")
    assert predicted is None
    assert tie is True


def test_predict_pairwise_winner_missing_score_is_not_tie():
    predicted, tie = predict_pairwise_winner(None, 0.4, "a", "b")
    assert predicted is None
    assert tie is False


def test_predict_pairwise_winner_left():
    predicted, tie = predict_pairwise_winner(0.7, 0.2, "a", "b")
    assert predicted == "a"
    assert tie is False


def test_basic_ranking_metrics():
    assert reciprocal_rank([0, 1, 0]) == 0.5
    assert ndcg_at_k([1, 0, 0], 3) == 1.0


def test_clustered_bootstrap_resamples_all_rows_from_each_step():
    result = clustered_bootstrap_mean_ci(
        [0.0, 1.0, 0.0],
        ["step-a", "step-a", "step-b"],
        seed=42,
        draws=500,
    )

    assert result["n"] == 3
    assert result["cluster_count"] == 2
    assert result["cluster_unit"] == "step_id"
    assert result["mean"] == pytest.approx(1 / 3)
    # Resampling either two-row step always keeps its 0 and 1 observations together.
    assert result["lower"] == 0.0
    assert result["upper"] == 0.5


def test_clustered_bootstrap_is_reproducible_with_a_fixed_seed():
    arguments = {
        "values": [1.0, 0.5, 0.0, 1.0],
        "cluster_ids": ["step-b", "step-a", "step-b", "step-c"],
        "seed": 99,
        "draws": 200,
        "confidence": 0.9,
    }

    first = clustered_bootstrap_mean_ci(**arguments)
    second = clustered_bootstrap_mean_ci(**arguments)

    assert first == second
    assert first["draws"] == 200
    assert first["seed"] == 99
    assert first["confidence"] == 0.9


def test_clustered_bootstrap_empty_result_has_full_metadata():
    result = clustered_bootstrap_mean_ci([], [], seed=7, draws=10, confidence=0.8)

    assert result == {
        "n": 0,
        "cluster_count": 0,
        "mean": None,
        "lower": None,
        "upper": None,
        "confidence": 0.8,
        "draws": 10,
        "seed": 7,
        "cluster_unit": "step_id",
    }


@pytest.mark.parametrize(
    ("values", "cluster_ids", "draws", "confidence", "message"),
    [
        ([1.0], [], 10, 0.95, "equal lengths"),
        ([1.0], ["step-a"], 0, 0.95, "at least 1"),
        ([1.0], ["step-a"], 10, 1.0, "between 0 and 1"),
    ],
)
def test_clustered_bootstrap_validates_arguments(values, cluster_ids, draws, confidence, message):
    with pytest.raises(ValueError, match=message):
        clustered_bootstrap_mean_ci(
            values,
            cluster_ids,
            draws=draws,
            confidence=confidence,
        )
