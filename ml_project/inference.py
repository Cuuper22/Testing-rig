from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .preprocessing import ScalerParams, decode_tokens, encode_tokens, scale_features
from .registry import ModelRegistry


@dataclass
class LoadedModel:
    """Container with loaded model artefacts."""

    weights: list[float]
    bias: float
    scaler: ScalerParams


def load_latest_model(registry: ModelRegistry, run_name: str) -> LoadedModel:
    """Load the most recent model registered under ``run_name``."""

    run = registry.get_latest_run(run_name)
    if run is None:
        raise RuntimeError(f"No runs found for {run_name}")
    weights, bias, scaler = registry.load_model(run.run_id)
    return LoadedModel(weights=weights, bias=bias, scaler=scaler)


def predict(features: Sequence[Sequence[float]], model: LoadedModel) -> list[float]:
    """Generate predictions using the provided model."""

    scaled, _ = scale_features(features, params=model.scaler)
    return [
        sum(weight * value for weight, value in zip(model.weights, row)) + model.bias
        for row in scaled
    ]


def tokenise_text(
    text: str, token_to_id: dict[str, int]
) -> tuple[list[int], dict[str, int]]:
    """Tokenise text and return encoded ids."""

    tokens = text.split()
    encoded = encode_tokens(tokens, token_to_id)
    return encoded, token_to_id


def detokenise(ids: Sequence[int], token_to_id: dict[str, int]) -> str:
    """Convert token identifiers back to their string form."""

    tokens = decode_tokens(list(ids), token_to_id)
    return " ".join(tokens)
