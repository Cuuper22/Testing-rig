import shutil
from pathlib import Path

import pytest

from ml_project.dataset import generate_synthetic_regression
from ml_project.inference import (
    LoadedModel,
    detokenise,
    load_latest_model,
    predict,
    tokenise_text,
)
from ml_project.preprocessing import build_tokenizer
from ml_project.registry import ModelRegistry
from ml_project.training import TrainingConfig, train_model


@pytest.fixture(autouse=True)
def clean_registry():
    path = Path("mlruns")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    yield
    shutil.rmtree(path)


def test_predict_after_training():
    dataset, _ = generate_synthetic_regression(60, 4, seed=321)
    registry = ModelRegistry()
    config = TrainingConfig(learning_rate=0.05, epochs=80, batch_size=15)
    train_model(dataset, config, registry, run_name="inference")

    loaded = load_latest_model(registry, "inference")
    assert isinstance(loaded, LoadedModel)

    predictions = predict(dataset.features, loaded)
    assert len(predictions) == len(dataset.labels)
    assert all(isinstance(value, float) for value in predictions)


def test_tokenise_and_detokenise_round_trip():
    tokenizer = build_tokenizer(["quick", "brown", "fox"])
    encoded, vocab = tokenise_text("quick brown", tokenizer)
    decoded = detokenise(encoded, vocab)
    assert decoded == "quick brown"
