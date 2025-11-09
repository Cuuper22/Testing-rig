"""Utilities for selecting pages for annotation via active learning.

This module implements uncertainty-based scoring functions that can be
used to prioritize manuscript pages for human review. The predictions are
expected to be dictionaries containing metadata such as the page id,
confidence scores, and token-level probabilities. The functions are
pure-Python and do not require external numerical dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence

PredictionLike = MutableMapping[str, object]


@dataclass
class UncertaintyScore:
    """Container describing a scored prediction.

    Attributes
    ----------
    page_id:
        Identifier of the manuscript page.
    score:
        The computed uncertainty score where larger values indicate a
        higher priority for review.
    metadata:
        Arbitrary metadata pulled from the original prediction object.
    """

    page_id: str
    score: float
    metadata: Dict[str, object]


def _normalize_probabilities(probs: Sequence[float]) -> List[float]:
    """Normalize a sequence of non-negative values into probabilities."""

    total = float(sum(probs))
    if total <= 0:
        # When no information is available, return a uniform distribution
        # to avoid crashing downstream consumers.
        length = len(probs)
        return [1.0 / length for _ in probs] if length else []
    return [float(p) / total for p in probs]


def _shannon_entropy(probs: Sequence[float]) -> float:
    """Compute the Shannon entropy in bits for a probability distribution."""

    normalized = _normalize_probabilities(probs)
    entropy = 0.0
    for prob in normalized:
        if prob > 0:
            entropy -= prob * log2(prob)
    return entropy


def _least_confidence(probabilities: Sequence[float]) -> float:
    """Return the least confidence score (1 - max probability)."""

    normalized = _normalize_probabilities(probabilities)
    if not normalized:
        return 1.0
    return 1.0 - max(normalized)


def _margin_confidence(probabilities: Sequence[float]) -> float:
    """Return the margin sampling score (difference between top two probs)."""

    normalized = sorted(_normalize_probabilities(probabilities), reverse=True)
    if not normalized:
        return 1.0
    if len(normalized) == 1:
        return 1.0 - normalized[0]
    return 1.0 - (normalized[0] - normalized[1])


def _collect_probabilities(prediction: PredictionLike) -> Sequence[float]:
    """Extract a probability-like sequence from a prediction.

    The function looks for a ``probabilities`` field first. If not found,
    it falls back to ``token_confidences`` (assuming values within [0, 1])
    and finally to a scalar ``confidence`` field which is transformed into
    a two-class distribution of ``[confidence, 1 - confidence]``.
    """

    if "probabilities" in prediction:
        probs = prediction["probabilities"]
        if isinstance(probs, Sequence):
            return [float(p) for p in probs]
    if "token_confidences" in prediction:
        probs = prediction["token_confidences"]
        if isinstance(probs, Sequence):
            return [float(p) for p in probs]
    if "confidence" in prediction:
        conf = float(prediction["confidence"])  # type: ignore[arg-type]
        conf = min(max(conf, 0.0), 1.0)
        return [conf, 1.0 - conf]
    return []


def _score_prediction(prediction: PredictionLike, strategy: str) -> float:
    """Compute an uncertainty score for a prediction using a strategy."""

    strategy = strategy.lower()
    if strategy not in {"entropy", "least_confident", "margin", "min_token"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    probabilities = _collect_probabilities(prediction)

    if strategy == "entropy":
        return _shannon_entropy(probabilities)
    if strategy == "least_confident":
        return _least_confidence(probabilities)
    if strategy == "margin":
        return _margin_confidence(probabilities)
    # strategy == "min_token"
    if probabilities:
        return 1.0 - min(_normalize_probabilities(probabilities))
    return 1.0


def score_predictions(
    predictions: Iterable[PredictionLike], strategy: str = "entropy"
) -> List[UncertaintyScore]:
    """Score predictions according to the requested uncertainty strategy.

    Parameters
    ----------
    predictions:
        Iterable of prediction mappings. Each prediction must contain a
        ``page_id`` key.
    strategy:
        Name of the uncertainty strategy. Supported values are
        ``"entropy"``, ``"least_confident"``, ``"margin"``, and
        ``"min_token"``.

    Returns
    -------
    list of :class:`UncertaintyScore`
        The scored predictions sorted by decreasing uncertainty.
    """

    scored: List[UncertaintyScore] = []
    for prediction in predictions:
        page_id = str(prediction.get("page_id", ""))
        if not page_id:
            raise KeyError("Each prediction must include a 'page_id' field")
        score = _score_prediction(prediction, strategy)
        scored.append(
            UncertaintyScore(
                page_id=page_id,
                score=score,
                metadata={key: prediction[key] for key in prediction if key != "page_id"},
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def prioritize_samples(
    predictions: Iterable[PredictionLike],
    strategy: str = "entropy",
    limit: Optional[int] = None,
) -> List[PredictionLike]:
    """Return predictions sorted by decreasing uncertainty.

    Parameters
    ----------
    predictions:
        Iterable of prediction mappings.
    strategy:
        Uncertainty strategy to use. See :func:`score_predictions`.
    limit:
        Optional maximum number of predictions to return.
    """

    predictions_list = list(predictions)
    scored = score_predictions(predictions_list, strategy=strategy)
    lookup: Dict[str, PredictionLike] = {str(p["page_id"]): p for p in predictions_list}
    ordered: List[PredictionLike] = []
    for score in scored:
        prediction = dict(lookup[score.page_id])
        prediction["uncertainty_score"] = score.score
        ordered.append(prediction)

    if limit is not None:
        return ordered[:limit]
    return ordered


__all__ = ["UncertaintyScore", "prioritize_samples", "score_predictions"]
