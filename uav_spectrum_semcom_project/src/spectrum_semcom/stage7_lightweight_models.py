"""Leakage-safe lightweight risk models for Stage-7 S7.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spectrum_semcom.stage7_causal_baselines import dwell_bin


DERIVED_CATEGORICAL_NAMES = (
    "previous_symbol",
    "action_dwell_bin",
)


def model_matrix(
    dataset: dict,
    *,
    numeric_names: Sequence[str],
    categorical_names: Sequence[str],
) -> tuple[np.ndarray, int, tuple[str, ...]]:
    """Select registered features and append causal categorical variables."""

    base = np.asarray(dataset["features"], dtype=np.float64)
    base_names = tuple(dataset["feature_names"])
    if base.ndim != 2 or base.shape[1] != len(base_names):
        raise ValueError("invalid causal feature matrix")
    available = {
        name: base[:, position]
        for position, name in enumerate(base_names)
    }
    available["previous_symbol"] = np.asarray(
        dataset["previous_symbol"], dtype=np.float64
    )
    available["action_dwell_bin"] = dwell_bin(
        np.asarray(dataset["action_dwell_scenes"], dtype=np.int64)
    ).astype(np.float64)
    numeric = tuple(numeric_names)
    categorical = tuple(categorical_names)
    requested = numeric + categorical
    if (
        len(set(requested)) != len(requested)
        or any(name not in available for name in requested)
        or "site" in requested
        or "group" in requested
    ):
        raise ValueError("invalid or unavailable registered model feature")
    matrix = np.column_stack([available[name] for name in requested])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("model features must be finite")
    return matrix, len(numeric), requested


def build_logistic_pipeline(
    *,
    numeric_count: int,
    total_count: int,
    regularization_c: float,
    class_weight: str | None,
    solver: str = "liblinear",
    maximum_iterations: int = 2000,
    random_seed: int = 0,
) -> Pipeline:
    """Create a train-only standardization and categorical one-hot pipeline."""

    numeric_indices = list(range(int(numeric_count)))
    categorical_indices = list(
        range(int(numeric_count), int(total_count))
    )
    if (
        not numeric_indices
        or not categorical_indices
        or float(regularization_c) <= 0.0
        or class_weight not in (None, "balanced")
    ):
        raise ValueError("invalid logistic pipeline specification")
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_indices),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_indices,
            ),
        ]
    )
    classifier = LogisticRegression(
        C=float(regularization_c),
        class_weight=class_weight,
        solver=solver,
        max_iter=int(maximum_iterations),
        random_state=int(random_seed),
    )
    return Pipeline(
        [("features", transform), ("classifier", classifier)]
    )


def expand_discrete_survival_rows(
    matrix: np.ndarray,
    time_to_failure: np.ndarray,
    *,
    maximum_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand source scenes into pooled-logistic person-period hazard rows."""

    features = np.asarray(matrix, dtype=np.float64)
    times = np.asarray(time_to_failure, dtype=np.int64).reshape(-1)
    horizon = int(maximum_horizon)
    if (
        features.ndim != 2
        or features.shape[0] != times.size
        or horizon < 1
    ):
        raise ValueError("invalid discrete survival inputs")
    rows = []
    labels = []
    source_indices = []
    for source, event_time in enumerate(times):
        observed_event = 1 <= int(event_time) <= horizon
        periods = int(event_time) if observed_event else horizon
        for offset in range(1, periods + 1):
            rows.append(
                np.concatenate(
                    (features[source], np.asarray([float(offset)]))
                )
            )
            labels.append(int(observed_event and offset == int(event_time)))
            source_indices.append(source)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.uint8),
        np.asarray(source_indices, dtype=np.int64),
    )


@dataclass(frozen=True)
class FittedRiskModel:
    """One direct binary model or one pooled discrete-hazard model."""

    kind: str
    pipeline: Pipeline
    numeric_count: int
    maximum_horizon: int

    def predict_risk(
        self, matrix: np.ndarray, horizons: Sequence[int]
    ) -> np.ndarray:
        features = np.asarray(matrix, dtype=np.float64)
        requested = tuple(int(value) for value in horizons)
        if (
            features.ndim != 2
            or not requested
            or any(value < 1 for value in requested)
            or any(value > self.maximum_horizon for value in requested)
        ):
            raise ValueError("invalid risk prediction request")
        if self.kind == "direct":
            if len(requested) != 1:
                raise ValueError("direct model predicts one registered horizon")
            probability = self.pipeline.predict_proba(features)[:, 1]
            return probability.reshape(-1, 1)
        if self.kind != "discrete_survival":
            raise ValueError("unknown fitted model kind")
        expanded = np.vstack(
            [
                np.column_stack(
                    (
                        features,
                        np.full(features.shape[0], float(offset)),
                    )
                )
                for offset in range(1, self.maximum_horizon + 1)
            ]
        )
        hazards = self.pipeline.predict_proba(expanded)[:, 1].reshape(
            self.maximum_horizon, features.shape[0]
        ).T
        survival = np.cumprod(1.0 - hazards, axis=1)
        return np.column_stack(
            [1.0 - survival[:, horizon - 1] for horizon in requested]
        )


def fit_direct_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    numeric_count: int,
    regularization_c: float,
    class_weight: str | None,
    maximum_iterations: int,
    random_seed: int,
) -> FittedRiskModel:
    features = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if features.shape[0] != target.size or np.unique(target).size < 2:
        raise ValueError("direct logistic fit requires two aligned classes")
    pipeline = build_logistic_pipeline(
        numeric_count=numeric_count,
        total_count=features.shape[1],
        regularization_c=regularization_c,
        class_weight=class_weight,
        maximum_iterations=maximum_iterations,
        random_seed=random_seed,
    )
    pipeline.fit(features, target)
    return FittedRiskModel("direct", pipeline, numeric_count, 1)


def fit_discrete_survival_logistic(
    matrix: np.ndarray,
    time_to_failure: np.ndarray,
    *,
    numeric_count: int,
    maximum_horizon: int,
    regularization_c: float,
    class_weight: str | None,
    maximum_iterations: int,
    random_seed: int,
) -> FittedRiskModel:
    expanded, labels, _ = expand_discrete_survival_rows(
        matrix,
        time_to_failure,
        maximum_horizon=maximum_horizon,
    )
    if np.unique(labels).size < 2:
        raise ValueError("survival logistic fit requires observed events")
    pipeline = build_logistic_pipeline(
        numeric_count=numeric_count,
        total_count=expanded.shape[1],
        regularization_c=regularization_c,
        class_weight=class_weight,
        maximum_iterations=maximum_iterations,
        random_seed=random_seed,
    )
    pipeline.fit(expanded, labels)
    return FittedRiskModel(
        "discrete_survival",
        pipeline,
        numeric_count,
        int(maximum_horizon),
    )
