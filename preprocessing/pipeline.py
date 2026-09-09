"""Deterministic RobustDRNet preprocessing and training-only augmentation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import PreprocessingConfig


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Preprocessing requires opencv-python-headless. Install "
            "preprocessing/requirements.txt in the project environment."
        ) from exc
    return cv2


def resize_rgb(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    cv2 = _require_cv2()
    width, height = size[1], size[0]
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def apply_clahe_rgb(
    image_rgb: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    """Apply CLAHE to LAB luminance.

    RobustDRNet specifies CLAHE but not its color space, clip limit, or tile
    grid. The choices are therefore explicit reproduction assumptions.
    """
    cv2 = _require_cv2()
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(image_lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def apply_circular_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Keep the maximum image-centred inscribed circle and zero its exterior.

    The paper specifies circular masking but does not define centre or radius.
    No retinal cropping, thresholding, contour detection, or other operation
    is performed.
    """
    height, width = image_rgb.shape[:2]
    centre_y = (height - 1) / 2.0
    centre_x = (width - 1) / 2.0
    radius = min(height, width) / 2.0
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - centre_x) ** 2 + (yy - centre_y) ** 2 <= radius**2
    return np.where(mask[..., None], image_rgb, 0).astype(np.uint8)


def normalize_imagenet(
    image_rgb: np.ndarray,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return (image - mean_array) / std_array


def preprocess_image(
    image_rgb: np.ndarray,
    *,
    training: bool,
    config: PreprocessingConfig | None = None,
    augmentation: Any | None = None,
) -> np.ndarray:
    """Return a normalized CHW float32 image.

    Deterministic order:
      1. resize to 224 x 224 x 3
      2. CLAHE
      3. circular mask
      4. optional random augmentation for training only
      5. ImageNet normalization

    Validation and test calls must use training=False. Any supplied
    augmentation is rejected in that mode to prevent accidental leakage.
    """
    config = config or PreprocessingConfig()
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC image with 3 channels, got {image_rgb.shape}")
    if not training and augmentation is not None:
        raise ValueError("Random augmentation is forbidden for validation/test data")

    image = resize_rgb(image_rgb, config.image_size)
    image = apply_clahe_rgb(
        image,
        clip_limit=config.clahe_clip_limit,
        tile_grid_size=config.clahe_tile_grid_size,
    )
    image = apply_circular_mask(image)

    if training:
        if augmentation is None:
            from .augmentations import build_training_augmentation

            augmentation = build_training_augmentation(config)
        image = augmentation(image=image)["image"]

    image = normalize_imagenet(
        image,
        mean=config.imagenet_mean,
        std=config.imagenet_std,
    )
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
