"""Translation pipeline that post-processes OCR output before MT inference."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Regular expression capturing punctuation that should be normalized to ASCII
# equivalents. Coptic uses a mixture of Greek punctuation marks that do not
# always map to Unicode punctuation used during training.
PUNCTUATION_NORMALIZATION = {
    "·": ".",
    "·": ".",
    "—": "-",
    "–": "-",
    "“": '"',
    "”": '"',
    "‚": ",",
    "„": ",",
}


@dataclass
class PipelineConfig:
    model_path: str
    device: str | None = None
    num_beams: int = 4
    max_length: int = 256


class TranslationPipeline:
    """Simple wrapper around a seq2seq MT model for Coptic OCR output."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path).to(self.device)

    @staticmethod
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        # Remove diacritics and combining marks commonly introduced by OCR noise.
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        # Normalize punctuation.
        for src, dst in PUNCTUATION_NORMALIZATION.items():
            text = text.replace(src, dst)
        # Collapse whitespace and strip extra punctuation spaces.
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:?!])", r"\1", text)
        return text

    def _prepare_inputs(self, texts: Sequence[str]) -> List[str]:
        normalized = [self.normalize_text(text) for text in texts]
        return [text for text in normalized if text]

    def translate(self, texts: Sequence[str]) -> List[str]:
        processed = self._prepare_inputs(texts)
        if not processed:
            return []
        tokenized = self.tokenizer(
            processed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        generated = self.model.generate(
            **tokenized,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
        )
        outputs = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [output.strip() for output in outputs]

    def __call__(self, ocr_output: Iterable[str] | str) -> List[str]:
        if isinstance(ocr_output, str):
            texts = [ocr_output]
        else:
            texts = list(ocr_output)
        return self.translate(texts)


__all__ = ["PipelineConfig", "TranslationPipeline"]
