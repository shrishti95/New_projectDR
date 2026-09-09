"""Configuration with an explicit audit trail for underspecified parameters."""

from __future__ import annotations

from dataclasses import dataclass
import warnings


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# RobustDRNet states which operations were used but does not publish these
# numerical or implementation details. Values below are runnable reproduction
# assumptions, not values claimed by the paper.
UNDERSPECIFIED_ASSUMPTIONS = {
    "clahe_color_space": "LAB luminance channel",
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),
    "circular_mask_geometry": "image-centred maximum inscribed circle",
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "rotation_probability": 0.5,
    "rotation_limit_degrees": 30.0,
    "affine_probability": 0.5,
    "affine_scale_range": (0.9, 1.1),
    "affine_translation_fraction": (-0.1, 0.1),
    "brightness_contrast_probability": 0.5,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "coarse_dropout_probability": 0.25,
    "coarse_dropout_holes_range": (1, 8),
    "coarse_dropout_size_fraction": (0.05, 0.15),
    "augmentation_seed": 42,
}


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: tuple[int, int] = (224, 224)
    imagenet_mean: tuple[float, float, float] = IMAGENET_MEAN
    imagenet_std: tuple[float, float, float] = IMAGENET_STD

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotation_probability: float = 0.5
    rotation_limit_degrees: float = 30.0
    affine_probability: float = 0.5
    affine_scale_range: tuple[float, float] = (0.9, 1.1)
    affine_translation_fraction: tuple[float, float] = (-0.1, 0.1)
    brightness_contrast_probability: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    coarse_dropout_probability: float = 0.25
    coarse_dropout_holes_range: tuple[int, int] = (1, 8)
    coarse_dropout_size_fraction: tuple[float, float] = (0.05, 0.15)
    augmentation_seed: int = 42

    def warn_about_assumptions(self) -> None:
        details = "; ".join(
            f"{name}={value}" for name, value in UNDERSPECIFIED_ASSUMPTIONS.items()
        )
        warnings.warn(
            "RobustDRNet leaves these preprocessing/augmentation details "
            f"underspecified. Reproduction assumptions in use: {details}",
            UserWarning,
            stacklevel=2,
        )
