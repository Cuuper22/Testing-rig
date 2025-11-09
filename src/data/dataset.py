"""PyTorch dataset integrating Albumentations-based augmentations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import AugmentationPipeline, AugmentationConfig, build_augmentation_pipeline, set_global_seed

Tokenizer = Callable[[str], Sequence[int]]
@dataclass
class Sample:
    """Container describing a dataset sample."""

    image_path: Path
    transcription: str


class OCRDataset(Dataset):
    """Dataset yielding image tensors and transcription token tensors.

    Parameters
    ----------
    samples:
        Sequence of :class:`Sample` describing where images live and their
        associated transcription text.
    tokenizer:
        Callable converting transcription strings into a sequence of token
        identifiers. The resulting sequence will be converted to a
        ``torch.LongTensor``.
    augmentation_config:
        Optional :class:`AugmentationConfig` instance describing
        augmentations to be applied. If omitted only tensor conversion and
        normalization from the configuration defaults will be applied.
    tokenizer_pad_id:
        Optional integer used to pad empty token sequences.
    seed:
        Random seed ensuring deterministic augmentations.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        tokenizer: Tokenizer,
        augmentation_config: Optional[AugmentationConfig] = None,
        tokenizer_pad_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            set_global_seed(seed)
            random.seed(seed)

        self.samples: List[Sample] = list(samples)
        if not self.samples:
            raise ValueError("OCRDataset requires at least one sample")

        self.tokenizer = tokenizer
        self.tokenizer_pad_id = tokenizer_pad_id

        if augmentation_config is None:
            augmentation_config = AugmentationConfig()
        self.pipeline: AugmentationPipeline = build_augmentation_pipeline(
            augmentation_config
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = list(self.tokenizer(text))
        if not tokens and self.tokenizer_pad_id is not None:
            tokens = [self.tokenizer_pad_id]
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image_array = self._load_image(sample.image_path)
        image_tensor = self.pipeline(image_array)
        token_tensor = self._tokenize(sample.transcription)
        return image_tensor, token_tensor
