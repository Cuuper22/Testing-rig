import shutil
from pathlib import Path

import pytest

from ml_project.dataset import generate_synthetic_regression
from ml_project.preprocessing import build_tokenizer
from ml_project.registry import ModelRegistry
from ml_project.training import TrainingConfig, evaluate_model, train_model


@pytest.fixture(autouse=True)
def clean_registry():
    mlruns_path = Path("mlruns")
    if mlruns_path.exists():
        shutil.rmtree(mlruns_path)
    mlruns_path.mkdir()
    yield
    shutil.rmtree(mlruns_path)


def test_train_model_logs_to_registry():
    dataset, _ = generate_synthetic_regression(80, 5, seed=123)
    registry = ModelRegistry()
    config = TrainingConfig(learning_rate=0.05, epochs=50, batch_size=16)
    tokenizer = build_tokenizer(["hello", "world"])

    result = train_model(
        dataset,
        config,
        registry,
        run_name="demo",
        tokenizer=tokenizer,
    )

    assert result.metrics["mse"] >= 0.0
    model_file = Path("mlruns") / result.run_id / "model.json"
    assert model_file.exists()
    assert registry.load_metrics(result.run_id)["mse"] == pytest.approx(
        result.metrics["mse"]
    )
    assert registry.load_tokenizer(result.run_id) == tokenizer


def test_evaluate_model_matches_training_metric():
    dataset, _ = generate_synthetic_regression(40, 3, seed=0)
    registry = ModelRegistry()
    config = TrainingConfig(learning_rate=0.05, epochs=120, batch_size=20)

    result = train_model(dataset, config, registry, run_name="evaluation")
    metrics = evaluate_model(dataset, result.weights, result.bias, result.scaler)

    assert pytest.approx(metrics["mse"], rel=1e-3) == result.metrics["mse"]
