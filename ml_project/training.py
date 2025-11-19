from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .dataset import Dataset
from .preprocessing import ScalerParams, scale_features
from .registry import ModelRegistry


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    learning_rate: float = 0.05
    epochs: int = 200
    batch_size: int = 16


@dataclass
class TrainingResult:
    """Result of a training run."""

    weights: list[float]
    bias: float
    metrics: dict[str, float]
    scaler: ScalerParams
    run_id: str


def _initialise_parameters(n_features: int) -> tuple[list[float], float]:
    # Deterministic initialisation helps with reproducibility during testing.
    weights = [0.0 for _ in range(n_features)]
    bias = 0.0
    return weights, bias


def _dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def train_model(
    dataset: Dataset,
    config: TrainingConfig,
    registry: ModelRegistry,
    run_name: str,
    tokenizer: dict[str, int] | None = None,
) -> TrainingResult:
    """Train a linear regression model and log artefacts to the registry."""

    features_scaled, scaler = scale_features(dataset.features)
    scaled_dataset = Dataset(features_scaled, dataset.labels)

    weights, bias = _initialise_parameters(len(features_scaled[0]))

    for _ in range(config.epochs):
        for batch_features, batch_labels in scaled_dataset.batches(config.batch_size):
            grad_w = [0.0 for _ in range(len(weights))]
            grad_b = 0.0

            for feature_row, label in zip(batch_features, batch_labels):
                prediction = _dot_product(feature_row, weights) + bias
                error = prediction - label
                for idx, value in enumerate(feature_row):
                    grad_w[idx] += value * error
                grad_b += error

            batch_size = len(batch_features)
            if batch_size == 0:
                continue
            for idx in range(len(weights)):
                weights[idx] -= config.learning_rate * (grad_w[idx] / batch_size)
            bias -= config.learning_rate * (grad_b / batch_size)

    predictions = [
        _dot_product(feature_row, weights) + bias for feature_row in features_scaled
    ]
    squared_errors = [
        (prediction - label) ** 2
        for prediction, label in zip(predictions, dataset.labels)
    ]
    mse = sum(squared_errors) / len(dataset.labels)

    run_id = registry.start_run(run_name)
    registry.log_metrics(run_id, {"mse": mse})
    registry.log_model(run_id, weights, bias, scaler)

    if tokenizer is not None:
        registry.log_tokenizer(run_id, tokenizer)

    return TrainingResult(
        weights=weights,
        bias=bias,
        metrics={"mse": mse},
        scaler=scaler,
        run_id=run_id,
    )


def evaluate_model(
    dataset: Dataset, weights: Sequence[float], bias: float, scaler: ScalerParams
) -> dict[str, float]:
    """Evaluate the trained model using the provided scaler parameters."""

    scaled_features, _ = scale_features(dataset.features, params=scaler)
    predictions = [_dot_product(row, weights) + bias for row in scaled_features]
    squared_errors = [
        (prediction - label) ** 2
        for prediction, label in zip(predictions, dataset.labels)
    ]
    mse = sum(squared_errors) / len(dataset.labels)
    return {"mse": mse}
