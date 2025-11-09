"""Prediction pipeline for the Coptic OCR and translation models.

This module implements a light-weight inference stack that is able to:

* discover the most recent checkpoint for both the OCR and translation models;
* preprocess the input image prior to inference;
* run OCR on the image to extract Coptic text line-by-line together with a confidence
  score for each line; and
* translate the recognised lines into English, again providing confidences.

The pipeline is intentionally pragmatic – rather than depending on heavyweight
runtime requirements the module gracefully falls back to rule-based heuristics if a
real model or its checkpoints are not available.  This keeps the inference flow
fully functional in constrained environments while still exposing the same
interface that a production system would use.

The main entrypoint is :func:`predict_image`, however the module also exposes a
CLI via ``python -m src.inference.predict`` and is imported by the FastAPI
service declared in ``src/server/app.py``.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

try:
    from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat
except ImportError as exc:  # pragma: no cover - pillow is expected but we fail loudly.
    raise RuntimeError(
        "Pillow is required to run inference. Please install it before executing the pipeline."
    ) from exc


LOGGER = logging.getLogger(__name__)


@dataclass
class LinePrediction:
    """Representation of a single line predicted by the OCR or translator."""

    text: str
    confidence: float

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return {"text": self.text, "confidence": round(float(self.confidence), 4)}


@dataclass
class PredictionResult:
    """Full output returned by the prediction pipeline."""

    coptic: List[LinePrediction]
    translation: List[LinePrediction]

    def to_dict(self) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        return {
            "coptic": [line.to_dict() for line in self.coptic],
            "translation": [line.to_dict() for line in self.translation],
        }


class CheckpointRegistry:
    """Utility that discovers the latest ("best") checkpoint for a model.

    The registry expects the following directory structure by default::

        checkpoints/
            ocr/
                best.ckpt
            translator/
                best.ckpt

    Any ``*.ckpt`` or ``*.pt`` file whose filename contains ``"best"`` will be
    considered a candidate.  When multiple matches exist the one with the most
    recent modification time is returned.  If no checkpoint exists a ``None``
    value is returned to signal that the heuristics-based fallback should be used.
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)

    def find_best(self, model: str) -> Optional[Path]:
        model_dir = self.root / model
        if not model_dir.exists():
            LOGGER.warning("Checkpoint directory for model '%s' does not exist: %s", model, model_dir)
            return None

        candidates = sorted(
            [
                path
                for path in model_dir.iterdir()
                if path.suffix in {".pt", ".pth", ".ckpt"} and "best" in path.stem.lower()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            LOGGER.warning("No best checkpoint found for model '%s' in %s", model, model_dir)
            return None

        best = candidates[0]
        LOGGER.info("Using checkpoint for %s: %s", model, best)
        return best


class ImagePreprocessor:
    """Apply light-weight preprocessing to improve OCR results."""

    def __init__(self, resize_height: int = 1280, adaptive_threshold: bool = True):
        self.resize_height = resize_height
        self.adaptive_threshold = adaptive_threshold

    def __call__(self, image: Image.Image) -> Image.Image:
        LOGGER.debug("Original image size: {0}x{1}".format(*image.size))

        if self.resize_height and image.height > self.resize_height:
            ratio = self.resize_height / float(image.height)
            new_size = (int(image.width * ratio), self.resize_height)
            image = image.resize(new_size, Image.BICUBIC)
            LOGGER.debug("Resized image to: {0}x{1}".format(*image.size))

        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.MedianFilter(size=3))

        if self.adaptive_threshold:
            image = self._adaptive_threshold(image)

        return image

    @staticmethod
    def _adaptive_threshold(image: Image.Image, block_size: int = 15, offset: int = 7) -> Image.Image:
        """Apply a simplified adaptive threshold using local averaging.

        The implementation uses a Gaussian blur as a fast approximation of the
        local neighbourhood mean and then thresholds the difference between the
        original image and the blurred representation.
        """

        radius = max(1, block_size // 2)
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        difference = ImageChops.subtract(image, blurred)
        lut = [255 if value > offset else 0 for value in range(256)]
        return difference.point(lut)


class OCRModel:
    """Wrapper around the OCR inference logic.

    The implementation will attempt to use ``pytesseract`` if available. When it
    is not installed the model falls back to a simple heuristics-based reader
    which cannot decode characters but provides a deterministic placeholder
    output.  The fallback still produces confidences derived from the contrast
    level of the preprocessed image; this makes the API useful in automated tests
    and during development when OCR dependencies are unavailable.
    """

    def __init__(self, checkpoint: Optional[Path]):
        self.checkpoint = checkpoint
        try:
            import pytesseract  # type: ignore

            self._tesseract = pytesseract
            LOGGER.info("pytesseract successfully imported for OCR inference")
        except ImportError:
            self._tesseract = None
            LOGGER.warning(
                "pytesseract is not available; falling back to heuristic OCR. "
                "Install pytesseract for improved recognition accuracy."
            )

    def infer(self, image: Image.Image) -> List[LinePrediction]:
        if self._tesseract is not None:
            raw_text = self._tesseract.image_to_string(image, lang="eng")
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            predictions = [
                LinePrediction(text=line, confidence=0.9) for line in lines
            ]
            if not predictions:
                predictions = [LinePrediction(text="", confidence=0.0)]
            return predictions

        # Fallback heuristics: estimate pseudo text based on brightness.
        stat = ImageStat.Stat(image)
        brightness = stat.mean[0] if stat.mean else 0.0
        contrast = stat.stddev[0] if stat.stddev else 0.0
        LOGGER.debug("Fallback OCR statistics: brightness=%s, contrast=%s", brightness, contrast)

        pseudo_tokens = self._brightness_to_tokens(brightness, contrast)
        confidence = max(0.15, min(0.85, 0.4 + contrast / 128.0))
        return [LinePrediction(text=" ".join(pseudo_tokens), confidence=confidence)]

    @staticmethod
    def _brightness_to_tokens(brightness: float, contrast: float) -> List[str]:
        alphabet = ["ⲁ", "ⲃ", "ⲅ", "ⲇ", "ⲉ", "ⲋ", "ⲍ", "ⲏ", "ⲑ", "ⲓ"]
        index = int(brightness / 255.0 * (len(alphabet) - 1)) if brightness else 0
        token = alphabet[index]
        repetitions = max(1, int(math.ceil(max(contrast, 1.0) / 10.0)))
        return [token] * repetitions


class TranslationModel:
    """Simple dictionary-based translation model.

    The translator loads a mapping from the checkpoint file when one is present.
    Checkpoints are expected to be stored in JSON format containing ``{"coptic":
    "english"}`` pairs.  When the checkpoint is missing, a built-in lexicon is
    used as a sensible default.
    """

    DEFAULT_LEXICON = {
        "ⲁ": "the",
        "ⲃ": "of",
        "ⲅ": "and",
        "ⲇ": "god",
        "ⲉ": "is",
        "ⲓ": "light",
    }

    def __init__(self, checkpoint: Optional[Path]):
        self.lexicon = dict(self.DEFAULT_LEXICON)
        if checkpoint is not None and checkpoint.exists():
            try:
                self.lexicon.update(json.loads(checkpoint.read_text(encoding="utf8")))
                LOGGER.info("Loaded translation lexicon from %s", checkpoint)
            except json.JSONDecodeError as exc:
                LOGGER.error("Failed to parse translation checkpoint %s: %s", checkpoint, exc)

    def translate(self, lines: Sequence[LinePrediction]) -> List[LinePrediction]:
        translated: List[LinePrediction] = []
        for line in lines:
            words = self._tokenise(line.text)
            translated_words = [self.lexicon.get(word, word) for word in words]
            confidence = min(0.99, max(0.2, line.confidence + 0.05))
            translated.append(LinePrediction(" ".join(translated_words), confidence))
        return translated

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        text = text.replace("\n", " ")
        return [token for token in text.split(" ") if token]


class PredictionPipeline:
    """High-level orchestration for the OCR and translation workflow."""

    def __init__(
        self,
        checkpoint_root: Union[str, Path] = "checkpoints",
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        self.checkpoints = CheckpointRegistry(checkpoint_root)
        self.preprocessor = preprocessor or ImagePreprocessor()
        self._ocr_model: Optional[OCRModel] = None
        self._translation_model: Optional[TranslationModel] = None

    def _ensure_models_loaded(self) -> None:
        if self._ocr_model is None:
            ocr_checkpoint = self.checkpoints.find_best("ocr")
            self._ocr_model = OCRModel(ocr_checkpoint)
        if self._translation_model is None:
            translation_checkpoint = self.checkpoints.find_best("translator")
            self._translation_model = TranslationModel(translation_checkpoint)

    def warm_up(self) -> None:
        """Load model artefacts into memory.

        The FastAPI application uses this hook to prepare the service during the
        readiness checks ensuring that the first prediction request does not pay
        the model loading cost.
        """

        self._ensure_models_loaded()

    def predict(self, image_input: Union[str, Path, bytes, Image.Image, io.BytesIO]) -> PredictionResult:
        self._ensure_models_loaded()
        assert self._ocr_model is not None and self._translation_model is not None

        image = self._load_image(image_input)
        LOGGER.debug("Image loaded for prediction: size=%s", image.size)

        processed = self.preprocessor(image)
        LOGGER.debug("Image preprocessing complete")

        coptic_lines = self._ocr_model.infer(processed)
        LOGGER.debug("OCR produced %d lines", len(coptic_lines))

        translations = self._translation_model.translate(coptic_lines)
        LOGGER.debug("Translation produced %d lines", len(translations))

        return PredictionResult(coptic=coptic_lines, translation=translations)

    @staticmethod
    def _load_image(image_input: Union[str, Path, bytes, Image.Image, io.BytesIO]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input
        if isinstance(image_input, io.BytesIO):
            image_input.seek(0)
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB")


def predict_image(image: Union[str, Path, bytes, Image.Image, io.BytesIO]) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    """Convenience wrapper used by external callers.

    Parameters
    ----------
    image:
        Any object accepted by :meth:`PredictionPipeline.predict`.

    Returns
    -------
    dict
        JSON serialisable dictionary containing the Coptic OCR output and its
        translation.
    """

    pipeline = PredictionPipeline()
    result = pipeline.predict(image)
    return result.to_dict()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Coptic OCR prediction pipeline.")
    parser.add_argument(
        "--image",
        "-i",
        nargs="+",
        required=True,
        help="Path(s) to input image files to process.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints",
        help="Directory that contains the model checkpoints.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to write the aggregated JSON results to.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity of the logger.",
    )
    return parser.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    pipeline = PredictionPipeline(checkpoint_root=args.checkpoint_root)
    images: Sequence[str] = args.image
    total = len(images)

    aggregated: Dict[str, Dict[str, List[Dict[str, Union[str, float]]]]] = {}
    for index, image_path in enumerate(images, start=1):
        LOGGER.info("Processing image %s/%s: %s", index, total, image_path)
        try:
            result = pipeline.predict(image_path)
            aggregated[image_path] = result.to_dict()
            LOGGER.info("Finished image %s/%s", index, total)
        except Exception as exc:  # pragma: no cover - CLI level error handling.
            LOGGER.error("Failed to process %s: %s", image_path, exc)

    output_json = json.dumps(aggregated, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output_json, encoding="utf8")
        LOGGER.info("Results written to %s", args.output)
    else:
        print(output_json)

    return 0


if __name__ == "__main__":  # pragma: no cover - entrypoint
    raise SystemExit(main())
