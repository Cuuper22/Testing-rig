"""Image augmentation utilities using Albumentations and torchvision.

This module exposes helpers to build deterministic augmentation pipelines
for OCR-style datasets. Augmentations are powered by Albumentations while
final tensor conversions and optional post-processing are handled by
``torchvision.transforms``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms

__all__ = [
    "AugmentationConfig",
    "AugmentationPipeline",
    "build_augmentation_pipeline",
    "set_global_seed",
]


@dataclass
class AugmentationConfig:
    """Configuration describing the augmentation pipeline.

    Attributes
    ----------
    elastic_prob:
        Probability of applying elastic distortions via
        :class:`albumentations.ElasticTransform`.
    elastic_alpha:
        Scaling factor that controls the intensity of the elastic
        deformation.
    elastic_sigma:
        Gaussian kernel size used for the elastic deformation.
    elastic_alpha_affine:
        Intensity of the affine transformations used alongside the
        elastic deformation.
    noise_prob:
        Probability of applying Gaussian noise via
        :class:`albumentations.GaussNoise`.
    noise_var_limit:
        Tuple describing the variance range for Gaussian noise.
    brightness_contrast_prob:
        Probability of adjusting brightness and contrast using
        :class:`albumentations.RandomBrightnessContrast`.
    brightness_limit:
        Maximum delta for brightness adjustment.
    contrast_limit:
        Maximum delta for contrast adjustment.
    blur_prob:
        Probability of applying blur with
        :class:`albumentations.Blur`.
    blur_limit:
        Maximum kernel size for the blur operation.
    normalize_mean, normalize_std:
        Per-channel mean and standard deviation used for normalization.
        These statistics are forwarded to
        :class:`albumentations.Normalize`.
    additional_torchvision_transforms:
        Extra torchvision transforms applied after Albumentations has
        converted the image into a PyTorch tensor. This allows call sites
        to incorporate domain-specific augmentations such as random
        erasing.
    """

    elastic_prob: float = 0.2
    elastic_alpha: float = 40.0
    elastic_sigma: float = 6.0
    elastic_alpha_affine: float = 20.0

    noise_prob: float = 0.2
    noise_var_limit: Sequence[float] = (10.0, 50.0)

    brightness_contrast_prob: float = 0.2
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2

    blur_prob: float = 0.1
    blur_limit: int = 3

    normalize_mean: Sequence[float] = (0.5, 0.5, 0.5)
    normalize_std: Sequence[float] = (0.5, 0.5, 0.5)

    additional_torchvision_transforms: Iterable[Callable] = field(
        default_factory=list
    )


class AugmentationPipeline:
    """Callable pipeline that applies image augmentations.

    The pipeline first executes the Albumentations sequence and then the
    optional torchvision transforms.
    """

    def __init__(self, albumentations_transform: A.BasicTransform,
                 torchvision_transforms: Optional[Callable] = None) -> None:
        self.albumentations_transform = albumentations_transform
        self.torchvision_transforms = torchvision_transforms

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        result = self.albumentations_transform(image=image)
        transformed: torch.Tensor = result["image"]
        if self.torchvision_transforms is not None:
            transformed = self.torchvision_transforms(transformed)
        return transformed


def build_augmentation_pipeline(config: AugmentationConfig) -> AugmentationPipeline:
    """Create an :class:`AugmentationPipeline` from a configuration.

    Parameters
    ----------
    config:
        Augmentation configuration controlling probabilities and
        magnitudes.
    """

    transforms_list: List[A.BasicTransform] = []

    if config.elastic_prob > 0:
        transforms_list.append(
            A.ElasticTransform(
                alpha=config.elastic_alpha,
                sigma=config.elastic_sigma,
                alpha_affine=config.elastic_alpha_affine,
                p=config.elastic_prob,
            )
        )

    if config.noise_prob > 0:
        transforms_list.append(
            A.GaussNoise(var_limit=tuple(config.noise_var_limit), p=config.noise_prob)
        )

    if config.brightness_contrast_prob > 0:
        transforms_list.append(
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                p=config.brightness_contrast_prob,
            )
        )

    if config.blur_prob > 0:
        transforms_list.append(
            A.Blur(blur_limit=config.blur_limit, p=config.blur_prob)
        )

    transforms_list.extend(
        [
            A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
            ToTensorV2(),
        ]
    )

    torchvision_transform = None
    additional = list(config.additional_torchvision_transforms)
    if additional:
        torchvision_transform = transforms.Compose(additional)

    composed = A.Compose(transforms_list)
    return AugmentationPipeline(composed, torchvision_transform)


def set_global_seed(seed: int) -> None:
    """Seed random number generators used throughout the data pipeline."""

    if seed < 0:
        raise ValueError("Seed must be non-negative")

    A.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
