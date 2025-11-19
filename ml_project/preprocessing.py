"""Data preprocessing utilities used throughout the project."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass
class ScalerParams:
    """Parameters used to scale features."""

    mean: list[float]
    std: list[float]


def scale_features(
    features: Sequence[Sequence[float]],
    params: ScalerParams | None = None,
    eps: float = 1e-8,
) -> tuple[list[list[float]], ScalerParams]:
    """Standardise *features* using the supplied ``params``.

    When ``params`` is ``None`` the function calculates the mean and standard
    deviation from ``features`` and returns them together with the scaled data.
    When ``params`` are provided, they are reused which makes this function
    suitable for both training and inference.
    """

    rows = [list(row) for row in features]
    if not rows or not rows[0]:
        raise ValueError("features must contain at least one row and column")
    n_cols = len(rows[0])
    if any(len(row) != n_cols for row in rows):
        raise ValueError("all feature rows must have the same length")

    if params is None:
        mean = [sum(col) / len(rows) for col in zip(*rows)]
        std = []
        for idx, col in enumerate(zip(*rows)):
            mu = mean[idx]
            variance = sum((value - mu) ** 2 for value in col) / len(rows)
            std.append(math.sqrt(variance))
    else:
        mean = list(params.mean)
        std = list(params.std)

    adjusted_std = [1.0 if abs(value) < eps else value for value in std]
    scaled = [
        [
            (value - mean[col_idx]) / adjusted_std[col_idx]
            for col_idx, value in enumerate(row)
        ]
        for row in rows
    ]
    return scaled, ScalerParams(mean=mean, std=adjusted_std)


def build_tokenizer(vocabulary: Iterable[str]) -> dict[str, int]:
    """Create a simple tokenizer dictionary from the provided vocabulary."""

    token_to_id: dict[str, int] = {}
    for token in vocabulary:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    return token_to_id


def encode_tokens(tokens: Sequence[str], token_to_id: Mapping[str, int]) -> list[int]:
    """Encode tokens into their corresponding identifiers."""

    encoded: list[int] = []
    for token in tokens:
        if token not in token_to_id:
            raise KeyError(f"Unknown token: {token}")
        encoded.append(token_to_id[token])
    return encoded


def decode_tokens(
    token_ids: Sequence[int], token_to_id: Mapping[str, int]
) -> list[str]:
    """Decode token identifiers back into token strings."""

    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return [id_to_token[idx] for idx in token_ids]
