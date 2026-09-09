"""Lazy APTOS reader that accesses images directly inside the Kaggle ZIP."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import os
from pathlib import Path
import zipfile

import numpy as np

from .augmentations import build_training_augmentation
from .config import PreprocessingConfig
from .pipeline import preprocess_image


@dataclass(frozen=True)
class AptosRecord:
    image_id: str
    diagnosis: int | None


def load_aptos_records(
    zip_path: str | Path,
    *,
    labelled: bool = True,
) -> list[AptosRecord]:
    csv_name = "train.csv" if labelled else "test.csv"
    with zipfile.ZipFile(zip_path) as archive:
        text = archive.read(csv_name).decode("utf-8-sig")

    records: list[AptosRecord] = []
    for row in csv.DictReader(io.StringIO(text)):
        diagnosis = int(row["diagnosis"]) if labelled else None
        records.append(AptosRecord(row["id_code"], diagnosis))
    return records


class AptosZipDataset:
    """PyTorch-compatible map-style dataset without a hard torch dependency.

    The default PyTorch DataLoader collate function converts returned NumPy
    arrays and integer labels into tensors. Each worker lazily opens its own
    ZIP handle; raw images are never extracted or duplicated.
    """

    def __init__(
        self,
        zip_path: str | Path,
        records: list[AptosRecord],
        *,
        training: bool,
        config: PreprocessingConfig | None = None,
        warn_about_assumptions: bool = True,
    ) -> None:
        self.zip_path = str(Path(zip_path).resolve())
        self.records = list(records)
        self.training = training
        self.config = config or PreprocessingConfig()
        self.augmentation = (
            build_training_augmentation(self.config) if training else None
        )
        self._archive: zipfile.ZipFile | None = None
        self._archive_pid: int | None = None
        if warn_about_assumptions:
            self.config.warn_about_assumptions()

    def __len__(self) -> int:
        return len(self.records)

    def _get_archive(self) -> zipfile.ZipFile:
        pid = os.getpid()
        if self._archive is None or self._archive_pid != pid:
            if self._archive is not None:
                self._archive.close()
            self._archive = zipfile.ZipFile(self.zip_path)
            self._archive_pid = pid
        return self._archive

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        folder = "train_images" if record.diagnosis is not None else "test_images"
        member = f"{folder}/{record.image_id}.png"
        encoded = self._get_archive().read(member)

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "Dataset decoding requires opencv-python-headless"
            ) from exc

        image_bgr = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image_bgr is None:
            raise ValueError(f"Could not decode ZIP member: {member}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = preprocess_image(
            image_rgb,
            training=self.training,
            config=self.config,
            augmentation=self.augmentation,
        )

        sample: dict[str, object] = {
            "image": image,
            "image_id": record.image_id,
        }
        if record.diagnosis is not None:
            sample["label"] = record.diagnosis
        return sample

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_archive"] = None
        state["_archive_pid"] = None
        return state

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
            self._archive_pid = None

    def __del__(self) -> None:
        self.close()
