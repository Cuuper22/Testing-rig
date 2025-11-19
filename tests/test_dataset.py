import pytest

from ml_project.dataset import (
    Dataset,
    generate_synthetic_regression,
    train_validation_split,
)


def test_dataset_batches_cover_all_samples():
    features = [[float(i), float(i + 10)] for i in range(10)]
    labels = [float(i) for i in range(10)]
    dataset = Dataset(features, labels)

    batches = list(dataset.batches(batch_size=3))
    reconstructed_features = [row for batch in batches for row in batch[0]]
    reconstructed_labels = [label for batch in batches for label in batch[1]]

    assert reconstructed_features == features
    assert reconstructed_labels == labels


def test_batches_invalid_batch_size():
    dataset, _ = generate_synthetic_regression(10, 2, seed=0)
    with pytest.raises(ValueError):
        list(dataset.batches(0))


def test_train_validation_split():
    dataset, _ = generate_synthetic_regression(50, 3, seed=42)
    train, valid = train_validation_split(dataset, validation_ratio=0.2, seed=0)
    assert len(train) + len(valid) == len(dataset)
    assert len(train) == 40
    assert len(valid) == 10


def test_generate_synthetic_regression_returns_weights():
    dataset, weights = generate_synthetic_regression(20, 4, seed=1)
    assert len(dataset.features) == 20
    assert len(dataset.features[0]) == 4
    assert len(weights) == 4
