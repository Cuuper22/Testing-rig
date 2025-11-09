"""Preprocessing utilities for document images.

This module offers reusable functions that can be imported as part of a
processing pipeline as well as a command-line entry point that can be invoked
with ``python -m src.data.preprocess``.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:  # pragma: no cover - exercised indirectly in the tests
    import cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV is optional
    cv2 = None  # type: ignore

try:  # pragma: no cover - exercised indirectly in the tests
    from skimage import color, exposure, filters, io, morphology, transform, util
except Exception:  # pragma: no cover - scikit-image is optional
    color = exposure = filters = io = morphology = transform = util = None  # type: ignore

LOGGER = logging.getLogger(__name__)
LOG_PATH = Path("logs") / "preprocess.log"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class PreprocessingError(RuntimeError):
    """Raised when the preprocessing pipeline cannot continue."""


@dataclass
class DeskewResult:
    """Container for deskewed image data."""

    image: np.ndarray
    angle: float


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Return an array converted to ``uint8`` in the range [0, 255]."""

    if image.dtype == np.uint8:
        return image

    min_val = float(image.min())
    max_val = float(image.max())
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image.astype(np.float32) - min_val) / (max_val - min_val)
    return (scaled * 255).clip(0, 255).astype(np.uint8)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or grayscale image to grayscale.

    Parameters
    ----------
    image:
        The input image array.
    Returns
    -------
    :class:`numpy.ndarray`
        Grayscale image in ``uint8`` format.
    """

    if image.ndim == 2:
        return _ensure_uint8(image)

    if image.ndim == 3 and image.shape[2] in {3, 4}:
        if cv2 is not None:
            conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(_ensure_uint8(image), conversion)
            return gray
        if color is not None:
            gray = color.rgb2gray(image[..., :3])
            return _ensure_uint8(gray)
        raise PreprocessingError("Neither OpenCV nor scikit-image is available for color conversion.")

    raise ValueError("Unsupported image shape for grayscale conversion.")


def adaptive_threshold(image: np.ndarray, block_size: int = 35, offset: float = 10) -> np.ndarray:
    """Apply adaptive thresholding to the provided image."""

    gray = convert_to_grayscale(image)

    if cv2 is not None:
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            max(3, block_size | 1),
            offset,
        )

    if filters is not None:
        image_float = util.img_as_float(gray) if util is not None else gray.astype(np.float32) / 255.0
        threshold = filters.threshold_local(image_float, block_size, offset=offset)
        binary = image_float > threshold
        return (binary * 255).astype(np.uint8)

    raise PreprocessingError("Adaptive thresholding requires OpenCV or scikit-image.")


def remove_background(image: np.ndarray, kernel_size: Tuple[int, int] = (15, 15)) -> np.ndarray:
    """Remove slowly varying backgrounds using morphological operations."""

    gray = convert_to_grayscale(image)

    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        normalized = cv2.normalize(gray - background, None, 0, 255, cv2.NORM_MINMAX)
        return normalized

    if morphology is not None and exposure is not None and util is not None:
        selem = morphology.rectangle(*kernel_size)
        background = morphology.opening(gray, selem)
        result = gray.astype(np.float32) - background.astype(np.float32)
        result = exposure.rescale_intensity(result, out_range=(0, 1))
        return util.img_as_ubyte(result)

    raise PreprocessingError("Background removal requires OpenCV or scikit-image.")


def _compute_skew_angle(thresholded: np.ndarray) -> float:
    """Estimate the skew angle of a thresholded image."""

    coords = np.column_stack(np.where(thresholded > 0))
    if coords.size == 0:
        return 0.0

    if cv2 is not None:
        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = rect[-1]
        if angle < -45:
            angle += 90
        return angle

    # PCA-based fallback when OpenCV is unavailable
    coords_centered = coords - coords.mean(axis=0)
    cov = np.cov(coords_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = np.rad2deg(np.arctan2(principal[0], principal[1]))
    # Normalize the angle into [-45, 45] to match OpenCV's semantics
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return angle


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image around its center by ``angle`` degrees."""

    if cv2 is not None:
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if transform is not None and util is not None:
        rotated = transform.rotate(image, angle, resize=False, mode="edge", preserve_range=True)
        return rotated.astype(image.dtype)

    raise PreprocessingError("Image rotation requires OpenCV or scikit-image.")


def deskew_image(image: np.ndarray) -> DeskewResult:
    """Deskew an image and return the rotated result and estimated angle."""

    gray = convert_to_grayscale(image)
    thresh = adaptive_threshold(gray)
    angle = _compute_skew_angle(thresh)
    rotated = _rotate_image(image, -angle)
    return DeskewResult(image=rotated, angle=angle)


def _read_image(path: Path) -> np.ndarray:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise PreprocessingError(f"Unable to read image: {path}")
        return image
    if io is not None:
        return io.imread(path)
    raise PreprocessingError("Image loading requires OpenCV or scikit-image.")


def _save_image(path: Path, image: np.ndarray) -> None:
    if cv2 is not None:
        if not cv2.imwrite(str(path), image):
            raise PreprocessingError(f"Unable to write image: {path}")
        return
    if io is not None:
        io.imsave(path, image)
        return
    raise PreprocessingError("Image saving requires OpenCV or scikit-image.")


def _iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def process_images(input_dir: Path, output_dir: Path) -> List[Path]:
    """Process all images in ``input_dir`` and write results to ``output_dir``."""

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    processed_paths: List[Path] = []
    for path in _iter_image_files(input_dir):
        LOGGER.info("Processing %s", path)
        image = _read_image(path)
        gray = convert_to_grayscale(image)
        cleaned = remove_background(gray)
        deskewed = deskew_image(cleaned)
        binary = adaptive_threshold(deskewed.image)
        relative = path.relative_to(input_dir)
        output_path = output_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_image(output_path, binary)
        processed_paths.append(output_path)
        LOGGER.info("Saved processed image to %s (angle=%.2f)", output_path, deskewed.angle)
    return processed_paths


def _configure_logging(verbose: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch preprocess document images.")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing raw images.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store processed images.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    LOGGER.info("Starting preprocessing: input=%s output=%s", args.input, args.output)
    try:
        processed = process_images(args.input, args.output)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Processing failed: %s", exc)
        return 1
    LOGGER.info("Finished preprocessing %d images.", len(processed))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
