"""Tests for the preprocessing utilities."""
import pytest

np = pytest.importorskip("numpy")

from src.data.preprocess import (
    adaptive_threshold,
    convert_to_grayscale,
    deskew_image,
    remove_background,
)

try:  # pragma: no cover - exercised indirectly
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:  # pragma: no cover - exercised indirectly
    from skimage import transform
except Exception:  # pragma: no cover - optional dependency
    transform = None  # type: ignore

if cv2 is None and transform is None:  # pragma: no cover - environment guard
    pytest.skip("Tests require either OpenCV or scikit-image", allow_module_level=True)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    if cv2 is not None:
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rotated = transform.rotate(image, angle, resize=False, mode="edge", preserve_range=True)
    return rotated.astype(image.dtype)


def test_convert_pipeline_preserves_shape() -> None:
    base = np.zeros((128, 128), dtype=np.uint8)
    base[32:96, 48:80] = 200
    rgb = np.dstack([base] * 3)

    gray = convert_to_grayscale(rgb)
    cleaned = remove_background(gray)
    binary = adaptive_threshold(cleaned)

    assert gray.shape == base.shape
    assert cleaned.shape == base.shape
    assert binary.shape == base.shape
    assert set(np.unique(binary)).issubset({0, 255})


def test_deskew_recovers_known_angle() -> None:
    base = np.zeros((200, 200), dtype=np.uint8)
    base[80:120, 40:160] = 255
    known_angle = 15.0
    rotated = _rotate(base, known_angle)

    result = deskew_image(rotated)

    assert result.image.shape == rotated.shape
    assert pytest.approx(known_angle, abs=1.5) == result.angle
