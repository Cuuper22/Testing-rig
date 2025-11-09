"""Run a lightweight training loop to exercise the pipeline end-to-end."""

from __future__ import annotations

from ml_project.dataset import generate_synthetic_regression
from ml_project.preprocessing import build_tokenizer
from ml_project.registry import ModelRegistry
from ml_project.training import TrainingConfig, train_model


def main() -> None:
    dataset, _ = generate_synthetic_regression(n_samples=100, n_features=6, seed=7)
    tokenizer = build_tokenizer(["demo", "preview", "training"])
    registry = ModelRegistry()
    config = TrainingConfig(learning_rate=0.05, epochs=60, batch_size=20)

    result = train_model(
        dataset,
        config,
        registry,
        run_name="preview",
        tokenizer=tokenizer,
    )
    print(f"Run {result.run_id} completed with MSE={result.metrics['mse']:.4f}")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
