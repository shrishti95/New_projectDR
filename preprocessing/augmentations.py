"""Training-only random augmentation from the RobustDRNet paper."""

from __future__ import annotations

from typing import Any

from .config import PreprocessingConfig


def build_training_augmentation(
    config: PreprocessingConfig,
) -> Any:
    """Return the paper-listed random transforms for training data only.

    The paper names horizontal/vertical flips, rotations, affine scaling and
    translation, brightness/contrast shifts, and coarse dropout. It does not
    provide exact probabilities or ranges; config.py labels the runnable
    assumptions used here.
    """
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "Training augmentation requires albumentations. Install "
            "preprocessing/requirements.txt in the project environment."
        ) from exc

    min_holes, max_holes = config.coarse_dropout_holes_range
    min_size, max_size = config.coarse_dropout_size_fraction

    return A.Compose(
        [
            A.HorizontalFlip(p=config.horizontal_flip_probability),
            A.VerticalFlip(p=config.vertical_flip_probability),
            A.Rotate(
                limit=config.rotation_limit_degrees,
                border_mode=0,
                fill=0,
                p=config.rotation_probability,
            ),
            A.Affine(
                scale=config.affine_scale_range,
                translate_percent={
                    "x": config.affine_translation_fraction,
                    "y": config.affine_translation_fraction,
                },
                rotate=0,
                border_mode=0,
                fill=0,
                p=config.affine_probability,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                p=config.brightness_contrast_probability,
            ),
            A.CoarseDropout(
                num_holes_range=(min_holes, max_holes),
                hole_height_range=(min_size, max_size),
                hole_width_range=(min_size, max_size),
                fill=0,
                p=config.coarse_dropout_probability,
            ),
        ],
        seed=config.augmentation_seed,
    )
