# RobustDRNet preprocessing

This module reads APTOS images lazily from:

    /workspace/aptos2019-blindness-detection.zip

It does not extract or duplicate the raw images.

## Paper-specified deterministic pipeline

For training, validation, and test images:

1. Decode as RGB.
2. Resize directly to 224 x 224 x 3.
3. Apply CLAHE for local contrast enhancement.
4. Apply a circular mask to remove peripheral non-retinal regions.
5. Normalize with ImageNet mean and standard deviation.

There is deliberately no retinal cropping, contour detection, threshold-based
border removal, denoising, sharpening, color normalization, or other extra
preprocessing.

Random augmentation is applied only when a dataset is constructed with
training=True. Validation and test paths reject supplied augmentation and are
deterministic.

## Explicitly underspecified by the paper

RobustDRNet does not publish:

- CLAHE color space, clip limit, or tile-grid size
- exact circular-mask centre or radius
- augmentation probabilities
- exact rotation distribution (the text mentions both +/-15, +/-30 and
  90-degree rotations)
- affine scale and translation ranges
- brightness and contrast ranges
- coarse-dropout size and count
- augmentation seed

Runnable assumptions are centralized in config.py and a warning listing them
is emitted when a dataset is created. They must not be described as original
paper parameters. Current assumptions are LAB-luminance CLAHE with clip limit
2.0 and an 8 x 8 grid, a centred maximum inscribed circle, and conventional
bounded augmentation settings.

## Usage

From /workspace/project:

    python -m preprocessing.smoke_test

The AptosZipDataset class is compatible with a PyTorch DataLoader without
requiring raw image extraction. Its output image is float32 CHW with shape
3 x 224 x 224. PyTorch default collation converts the NumPy arrays to tensors.

The module never creates train/validation folds implicitly. Split construction
will be handled separately so that random seed, stratification, and leakage
controls remain auditable.
