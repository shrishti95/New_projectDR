"""Export frozen ConvNeXt-Tiny train/validation features for fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from preprocessing import AptosZipDataset
from training.models import build_model
from training.train_benchmark import load_fold_records, make_loader


ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def extract(model, loader, device):
    model.eval()
    ids, labels, features = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features_batch = model.classifier[:-1](model.avgpool(model.features(images)))
        features_batch = torch.flatten(features_batch, 1).float().cpu().numpy()
        ids.extend(batch["image_id"])
        labels.append(batch["label"].numpy())
        features.append(features_batch)
    return {
        "sample_id": np.asarray(ids),
        "label": np.concatenate(labels),
        "feature": np.concatenate(features),
    }


def run_fold(fold: int, zip_path: Path, manifest: Path, output_root: Path, batch_size: int, workers: int):
    train_records, validation_records = load_fold_records(manifest, fold)
    checkpoint_path = ROOT / "training/runs/convnext_tiny" / f"fold_{fold}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model("convnext_tiny", pretrained=False, num_classes=5)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda")
    model = model.to(device)
    output = output_root / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("train", train_records), ("validation", validation_records)):
        dataset = AptosZipDataset(zip_path, records, training=False, warn_about_assumptions=False)
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False, workers=workers, seed=1042 + fold)
        arrays = extract(model, loader, device)
        if arrays["feature"].shape != (len(records), 768):
            raise ValueError(f"Unexpected ConvNeXt feature shape: {arrays['feature'].shape}")
        np.savez_compressed(output / f"{name}_features.npz", **arrays)
        dataset.close()
        print(f"EXPORTED fold={fold} split={name} samples={len(records)}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("/workspace/data/aptos2019-blindness-detection/aptos2019-blindness-detection.zip"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "training/splits/aptos_five_folds.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "training/fusion/convnext_tiny_features")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with bfloat16 support is required")
    for fold in args.folds:
        run_fold(fold, args.zip, args.manifest, args.output_root, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
