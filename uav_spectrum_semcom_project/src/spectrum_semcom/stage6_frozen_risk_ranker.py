"""Frozen one-step hazard ranker for Stage-6 reliability budgeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectrum_semcom.stage6_temporal_hazard import (
    build_temporal_hazard_dataset,
)


@dataclass(frozen=True)
class FrozenRiskRanking:
    selected_c: float
    scores: np.ndarray
    training_rows: np.ndarray
    evaluation_rows: np.ndarray
    thresholds_by_fraction: dict[float, float | None]
    candidates_by_fraction: dict[float, np.ndarray]


def _model(c_value: float, maximum_iterations: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(maximum_iterations),
            random_state=0,
        ),
    )


def _fit_group_oof(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    c_value: float,
    maximum_iterations: int,
) -> np.ndarray:
    predictions = np.full(labels.size, np.nan, dtype=np.float64)
    for held_group in sorted(set(groups.tolist())):
        validation = groups == held_group
        training = ~validation
        if np.unique(labels[training]).size < 2:
            raise ValueError("risk-ranker training fold lacks both classes")
        estimator = _model(c_value, maximum_iterations)
        estimator.fit(features[training], labels[training])
        predictions[validation] = estimator.predict_proba(
            features[validation]
        )[:, 1]
    if not np.all(np.isfinite(predictions)):
        raise ValueError("risk-ranker OOF predictions are incomplete")
    return predictions


def fit_frozen_risk_ranking(
    channel_power_dbm: np.ndarray,
    states: list,
    timestamps: np.ndarray,
    cluster_ids: np.ndarray,
    codebook,
    *,
    train_groups: Iterable[str],
    evaluation_groups: Iterable[str],
    selected_c: float,
    reservation_fractions: Iterable[float],
    maximum_iterations: int = 5000,
) -> FrozenRiskRanking:
    """Refit only the preregistered model at its frozen hyperparameter."""

    fractions = tuple(sorted({float(value) for value in reservation_fractions}))
    if (
        not fractions
        or fractions[0] < 0.0
        or fractions[-1] >= 1.0
        or float(selected_c) <= 0.0
    ):
        raise ValueError("invalid frozen risk-ranking configuration")
    dataset = build_temporal_hazard_dataset(
        channel_power_dbm,
        states,
        timestamps,
        cluster_ids,
        codebook,
    )
    features = np.asarray(dataset["features"], dtype=np.float64)
    labels = np.asarray(dataset["labels"], dtype=np.uint8)
    groups = np.asarray(dataset["groups"]).astype(str)
    source_indices = np.asarray(
        dataset["source_indices"], dtype=np.int64
    )
    training_rows = np.isin(groups, np.asarray(tuple(train_groups)).astype(str))
    evaluation_rows = np.isin(
        groups, np.asarray(tuple(evaluation_groups)).astype(str)
    )
    if (
        not np.any(training_rows)
        or not np.any(evaluation_rows)
        or np.any(training_rows & evaluation_rows)
        or np.unique(labels[training_rows]).size < 2
    ):
        raise ValueError("invalid risk-ranker group split")

    training_scores = _fit_group_oof(
        features[training_rows],
        labels[training_rows],
        groups[training_rows],
        c_value=float(selected_c),
        maximum_iterations=int(maximum_iterations),
    )
    estimator = _model(float(selected_c), int(maximum_iterations))
    estimator.fit(features[training_rows], labels[training_rows])
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    scores[training_rows] = training_scores
    scores[evaluation_rows] = estimator.predict_proba(
        features[evaluation_rows]
    )[:, 1]
    if not np.all(np.isfinite(scores[training_rows | evaluation_rows])):
        raise ValueError("risk scores are incomplete")

    thresholds: dict[float, float | None] = {}
    candidates: dict[float, np.ndarray] = {}
    for fraction in fractions:
        selected = np.zeros(len(states), dtype=bool)
        if fraction == 0.0:
            threshold = None
        else:
            threshold = float(
                np.quantile(
                    training_scores,
                    1.0 - fraction,
                    method="higher",
                )
            )
            for source, score, is_evaluation in zip(
                source_indices, scores, evaluation_rows
            ):
                if is_evaluation and float(score) >= threshold:
                    selected[int(source) + 1] = True
        thresholds[fraction] = threshold
        candidates[fraction] = selected

    for lower, upper in zip(fractions, fractions[1:]):
        if np.any(candidates[lower] & ~candidates[upper]):
            raise ValueError("risk candidate sets are not nested")
    return FrozenRiskRanking(
        selected_c=float(selected_c),
        scores=scores,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        thresholds_by_fraction=thresholds,
        candidates_by_fraction=candidates,
    )
