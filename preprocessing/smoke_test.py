"""Run deterministic and training-path checks against the real APTOS ZIP."""

from __future__ import annotations

import argparse
from collections import Counter
import warnings

import numpy as np

from preprocessing import AptosZipDataset, load_aptos_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        default="/workspace/aptos2019-blindness-detection.zip",
        help="Path to the original Kaggle archive",
    )
    args = parser.parse_args()

    records = load_aptos_records(args.zip, labelled=True)
    counts = Counter(record.diagnosis for record in records)
    assert len(records) == 3662, f"Expected 3662 train rows, got {len(records)}"

    with warnings.catch_warnings(record=True) as caught:
        validation = AptosZipDataset(
            args.zip,
            records[:1],
            training=False,
        )
    first = validation[0]
    second = validation[0]
    image = first["image"]
    assert isinstance(image, np.ndarray)
    assert image.shape == (3, 224, 224)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert np.array_equal(first["image"], second["image"])
    validation.close()

    training = AptosZipDataset(
        args.zip,
        records[:1],
        training=True,
        warn_about_assumptions=False,
    )
    train_image = training[0]["image"]
    assert isinstance(train_image, np.ndarray)
    assert train_image.shape == (3, 224, 224)
    assert train_image.dtype == np.float32
    assert np.isfinite(train_image).all()
    training.close()

    print(f"train_records={len(records)}")
    print("class_counts=" + ",".join(f"{grade}:{counts[grade]}" for grade in range(5)))
    print(f"validation_shape={image.shape}")
    print(f"validation_range=({image.min():.6f},{image.max():.6f})")
    print("validation_is_deterministic=True")
    print("training_augmentation_path=True")
    print(f"underspecification_warning_emitted={bool(caught)}")


if __name__ == "__main__":
    main()
