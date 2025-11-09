"""Utility functions for the training pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch


def _levenshtein_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute the Levenshtein edit distance between two token sequences.

    Parameters
    ----------
    reference: Sequence[str]
        The target sequence.
    hypothesis: Sequence[str]
        The predicted sequence.

    Returns
    -------
    int
        The number of single-token edits required to transform ``hypothesis``
        into ``reference``.
    """

    if len(reference) == 0:
        return len(hypothesis)
    if len(hypothesis) == 0:
        return len(reference)

    previous_row = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current_row = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ref_token != hyp_token)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calculate_cer(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Compute the character error rate (CER).

    The CER is defined as the total edit distance between predicted and
    reference character sequences divided by the number of reference
    characters.
    """

    total_distance = 0
    total_length = 0

    for pred, ref in zip(predictions, references):
        ref_chars = list(ref)
        hyp_chars = list(pred)
        total_distance += _levenshtein_distance(ref_chars, hyp_chars)
        total_length += max(len(ref_chars), 1)

    if total_length == 0:
        return 0.0
    return total_distance / total_length


def calculate_wer(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Compute the word error rate (WER)."""

    total_distance = 0
    total_length = 0

    for pred, ref in zip(predictions, references):
        ref_words = ref.split()
        hyp_words = pred.split()
        total_distance += _levenshtein_distance(ref_words, hyp_words)
        total_length += max(len(ref_words), 1)

    if total_length == 0:
        return 0.0
    return total_distance / total_length


@dataclass
class SchedulerConfig:
    """Configuration container for learning rate schedulers."""

    name: str
    params: Optional[Dict[str, float]] = None


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Optional[SchedulerConfig | Dict[str, object]],
    total_steps: Optional[int] = None,
):
    """Instantiate a learning rate scheduler for the provided optimizer.

    Parameters
    ----------
    optimizer:
        The optimizer instance to which the scheduler will be attached.
    scheduler_config:
        Either a :class:`SchedulerConfig` or a dictionary with ``name`` and
        ``params`` keys describing the scheduler to instantiate.
    total_steps:
        Total number of optimization steps. Required for certain schedulers
        such as warmup schedules.
    """

    if scheduler_config is None:
        return None

    if isinstance(scheduler_config, dict):
        scheduler_config = SchedulerConfig(
            name=scheduler_config.get("name", ""),
            params=scheduler_config.get("params") or {},
        )

    name = scheduler_config.name.lower()
    params = scheduler_config.params or {}

    if name == "cosine_with_warmup":
        warmup_steps = params.get("warmup_steps")
        if warmup_steps is None and total_steps is not None:
            warmup_ratio = params.get("warmup_ratio", 0.0)
            warmup_steps = int(total_steps * warmup_ratio)
        num_cycles = params.get("num_cycles", 0.5)
        from math import cos, pi
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - (warmup_steps or 0), 1) if total_steps else params.get("t_max", 50),
            eta_min=params.get("eta_min", 0.0),
        )
        if warmup_steps:
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                if total_steps:
                    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return max(0.0, 0.5 * (1.0 + cos(pi * num_cycles * progress)))
                return 1.0

            return LambdaLR(optimizer, lr_lambda)
        return scheduler

    if name == "linear_warmup":
        warmup_steps = params.get("warmup_steps")
        if warmup_steps is None and total_steps is not None:
            warmup_ratio = params.get("warmup_ratio", 0.1)
            warmup_steps = int(total_steps * warmup_ratio)
        final_lr_scale = params.get("final_lr_scale", 0.0)
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            if total_steps:
                progress = (step - (warmup_steps or 0)) / float(max(1, total_steps - (warmup_steps or 0)))
                return max(final_lr_scale, 1.0 - progress)
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    if name == "step":
        from torch.optim.lr_scheduler import StepLR

        return StepLR(
            optimizer,
            step_size=int(params.get("step_size", 10)),
            gamma=float(params.get("gamma", 0.1)),
        )

    if name == "exponential":
        from torch.optim.lr_scheduler import ExponentialLR

        return ExponentialLR(
            optimizer,
            gamma=float(params.get("gamma", 0.99)),
        )

    raise ValueError(f"Unsupported scheduler type: {scheduler_config.name}")


def get_precision_kwargs(precision_config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Translate a precision configuration dictionary into Trainer kwargs."""

    if not precision_config:
        return {}

    allowed_keys = {"precision", "amp_backend", "amp_level", "strategy"}
    return {k: v for k, v in precision_config.items() if k in allowed_keys and v is not None}


__all__ = [
    "SchedulerConfig",
    "calculate_cer",
    "calculate_wer",
    "get_scheduler",
    "get_precision_kwargs",
]
