from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Dataset:
    """A simple in-memory dataset wrapping features and labels."""

    features: list[list[float]]
    labels: list[float]

    def __post_init__(self) -> None:
        if not self.features or not self.features[0]:
            raise ValueError("features must contain at least one row and column")
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must have the same length")
        row_length = len(self.features[0])
        if any(len(row) != row_length for row in self.features):
            raise ValueError("all feature rows must have the same length")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.features)

    def batches(
        self, batch_size: int
    ) -> Iterator[tuple[list[list[float]], list[float]]]:
        """Iterate over the dataset in batches."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            end = min(start + batch_size, len(self))
            yield self.features[start:end], self.labels[start:end]


def train_validation_split(
    dataset: Dataset,
    validation_ratio: float,
    seed: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Split ``dataset`` into training and validation subsets."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    split_idx = int(len(dataset) * (1 - validation_ratio))
    train_indices = indices[:split_idx]
    valid_indices = indices[split_idx:]

    train_features = [dataset.features[idx] for idx in train_indices]
    train_labels = [dataset.labels[idx] for idx in train_indices]
    valid_features = [dataset.features[idx] for idx in valid_indices]
    valid_labels = [dataset.labels[idx] for idx in valid_indices]

    return (
        Dataset(train_features, train_labels),
        Dataset(valid_features, valid_labels),
    )


def generate_synthetic_regression(
    n_samples: int,
    n_features: int,
    noise: float = 0.1,
    seed: int | None = None,
) -> tuple[Dataset, list[float]]:
    """Generate a synthetic linear regression dataset and the ground truth weights."""

    rng = random.Random(seed)
    weights = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
    features: list[list[float]] = []
    labels: list[float] = []

    for _ in range(n_samples):
        row = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
        noise_value = rng.gauss(0.0, noise)
        label = sum(value * weight for value, weight in zip(row, weights)) + noise_value
        features.append(row)
        labels.append(label)

    return Dataset(features, labels), weights
