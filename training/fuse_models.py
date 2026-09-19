"""Leakage-free soft-voting fusion for ConvNeXt-Tiny, Swin V2, and DINOv2.

Each model must provide validation probabilities for the same five folds. The
script aligns rows by image ID before averaging, so differences in loader order
cannot silently corrupt the ensemble.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.proposed_stage1.convnextv2.metrics import compute_metrics


MODELS = ("convnext_tiny", "swinv2", "dinov2")
DEFAULT_PATTERNS = {
    "convnext_tiny": "training/runs/convnext_tiny/fold_{fold}/validation_predictions.npz",
    "swinv2": "training/proposed_stage1/swinv2/runs/fold_{fold}/validation_features.npz",
    "dinov2": "training/proposed_stage1/dinov2/runs/fold_{fold}/validation_features.npz",
}


def _read_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction archive: {path}. "
            "Restore the tracked Git LFS artifacts before running fusion."
        )
    with path.open("rb") as handle:
        prefix = handle.read(80)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the downloaded archive. "
            "Install Git LFS and run: git lfs pull"
        )
    with np.load(path, allow_pickle=False) as data:
        id_key = "sample_id" if "sample_id" in data.files else "image_id"
        label_key = "label" if "label" in data.files else "target"
        required = {id_key, label_key, "probability"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        return {"sample_id": data[id_key].copy(), "label": data[label_key].copy(), "probability": data["probability"].copy()}


def _sample_key(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_fold(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    data = _read_archive(path)
    ids = np.asarray([_sample_key(value) for value in data["sample_id"]])
    labels = np.asarray(data["label"], dtype=np.int64)
    probability = np.asarray(data["probability"], dtype=np.float64)
    if probability.shape != (len(ids), 5):
        raise ValueError(f"{path}: expected N x 5 probabilities, got {probability.shape}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate sample IDs")
    return {key: value for key, value in zip(ids, probability)}, ids, labels


def _parse_weights(value: str) -> np.ndarray:
    weights = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("--weights must contain three non-negative numbers")
    total = weights.sum()
    if total <= 0:
        raise ValueError("At least one fusion weight must be positive")
    return weights / total


def fuse_fold(paths: dict[str, Path], weights: np.ndarray) -> dict[str, object]:
    archives = {name: _load_fold(path)[0] for name, path in paths.items()}
    common = set.intersection(*(set(archive) for archive in archives.values()))
    if not common:
        raise ValueError("The three prediction archives have no common sample IDs")
    ordered_ids = sorted(common)
    labels = []
    probabilities = []
    for sample_id in ordered_ids:
        rows = [archives[name][sample_id] for name in MODELS]
        if not np.allclose(rows[0].sum(), 1.0, atol=1e-5):
            raise ValueError(f"Invalid probability row for sample {sample_id}")
        probabilities.append(sum(weight * row for weight, row in zip(weights, rows)))
    # Labels are checked across models after alignment.
    for name in MODELS:
        raw = _read_archive(paths[name])
        lookup = {_sample_key(i): int(y) for i, y in zip(raw["sample_id"], raw["label"])}
        labels.append(np.asarray([lookup[sample_id] for sample_id in ordered_ids]))
    if not all(np.array_equal(labels[0], other) for other in labels[1:]):
        raise ValueError("The three archives disagree on validation labels")
    return {
        "sample_id": np.asarray(ordered_ids),
        "label": labels[0],
        "probability": np.asarray(probabilities),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--weights", default="1,1,1", help="ConvNeXt,Swin,DINO weights")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()

    root = args.root.resolve()
    weights = _parse_weights(args.weights)
    output = (args.output or root / "training/fusion/convnext_swin_dinov2.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    folds = []
    all_ids = []
    all_labels = []
    all_probabilities = []
    for fold in args.folds:
        paths = {name: root / pattern.format(fold=fold) for name, pattern in DEFAULT_PATTERNS.items()}
        result = fuse_fold(paths, weights)
        metrics = compute_metrics(result["label"], result["probability"])
        folds.append({"fold": fold, "samples": len(result["label"]), "metrics": metrics})
        all_ids.append(result["sample_id"])
        all_labels.append(result["label"])
        all_probabilities.append(result["probability"])

    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    ids = np.concatenate(all_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("A sample occurs in more than one validation fold")
    result = {
        "models": list(MODELS),
        "weights": dict(zip(MODELS, weights.tolist())),
        "folds": folds,
        "samples": len(labels),
        "metrics": compute_metrics(labels, probabilities),
        "prediction_archive": str(output.with_suffix(".npz")),
    }
    np.savez_compressed(output.with_suffix(".npz"), sample_id=ids, label=labels, probability=probabilities)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
