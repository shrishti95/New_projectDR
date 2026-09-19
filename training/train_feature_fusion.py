"""Train a small ConvNeXt-Tiny + Swin V2 feature-fusion head."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.proposed_stage1.convnextv2.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]
SWIN_ROOT = ROOT / "training/proposed_stage1/swinv2/runs"
CONVNEXT_ROOT = ROOT / "training/fusion/convnext_tiny_features"


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in ("sample_id", "label", "feature")}
    if result["feature"].ndim != 2:
        raise ValueError(f"Invalid feature array: {path}")
    return result


def align(branches: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookups = []
    for branch in branches:
        ids = [str(value) for value in branch["sample_id"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample IDs in feature archive")
        lookups.append({sample_id: index for index, sample_id in enumerate(ids)})
    common = sorted(set.intersection(*(set(lookup) for lookup in lookups)))
    if not common:
        raise ValueError("Feature archives have no common sample IDs")
    indices = [[lookup[sample_id] for sample_id in common] for lookup in lookups]
    labels = [branch["label"][index] for branch, index in zip(branches, indices)]
    if not all(np.array_equal(labels[0], label) for label in labels[1:]):
        raise ValueError("Feature archives disagree on labels")
    features = np.concatenate(
        [branch["feature"][index].astype(np.float32) for branch, index in zip(branches, indices)],
        axis=1,
    )
    return np.asarray(common), labels[0].astype(np.int64), features


class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(1792),
            nn.Linear(1792, 512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, 5),
        )

    def forward(self, features):
        return self.net(features)


def split_inner(labels: np.ndarray, seed: int = 42, fraction: float = 0.1):
    rng = np.random.default_rng(seed)
    train, inner = [], []
    for grade in range(5):
        indices = np.flatnonzero(labels == grade)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        inner.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return np.asarray(train), np.asarray(inner)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    for features, target in loader:
        logits = model(features.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def run_fold(fold: int, epochs: int, batch_size: int, output_root: Path):
    train_conv = load_features(CONVNEXT_ROOT / f"fold_{fold}/train_features.npz")
    train_swin = load_features(SWIN_ROOT / f"fold_{fold}/train_features.npz")
    val_conv = load_features(CONVNEXT_ROOT / f"fold_{fold}/validation_features.npz")
    val_swin = load_features(SWIN_ROOT / f"fold_{fold}/validation_features.npz")
    _, train_labels, train_features = align([train_conv, train_swin])
    sample_ids, val_labels, val_features = align([val_conv, val_swin])
    train_indices, inner_indices = split_inner(train_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionHead().to(device)
    counts = np.bincount(train_labels[train_indices], minlength=5)
    weights = torch.tensor(len(train_indices) / (5 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_features[train_indices]), torch.from_numpy(train_labels[train_indices])), batch_size=batch_size, shuffle=True)
    inner_loader = DataLoader(TensorDataset(torch.from_numpy(train_features[inner_indices]), torch.from_numpy(train_labels[inner_indices])), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels)), batch_size=batch_size, shuffle=False)

    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for features, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(device)), target.to(device))
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(target); count += len(target)
        inner_target, inner_probability = predict(model, inner_loader, device)
        inner_loss = float(nn.CrossEntropyLoss()(model(torch.from_numpy(train_features[inner_indices]).to(device)), torch.from_numpy(train_labels[inner_indices]).to(device)).detach().cpu())
        row = {"epoch": epoch, "train_loss": total / count, "inner_loss": inner_loss, **compute_metrics(inner_target, inner_probability)}
        history.append(row)
        if inner_loss < best_loss:
            best_loss = inner_loss; stale = 0; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(best_state)
    target, probability = predict(model, val_loader, device)
    result = {"fold": fold, "samples": len(target), "feature_dimension": 1792, "best_inner_loss": best_loss, "metrics": compute_metrics(target, probability), "history": history, "weights": weights.cpu().tolist()}
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "fold": fold, "feature_dimension": 1792}, output_root / f"fold_{fold}.pt")
    np.savez_compressed(output_root / f"fold_{fold}_validation_predictions.npz", sample_id=sample_ids, label=target, probability=probability)
    (output_root / f"fold_{fold}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fold": fold, **result["metrics"]}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=ROOT / "training/fusion/convnext_swin_feature")
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    results = [run_fold(fold, args.epochs, args.batch_size, args.output_root) for fold in args.folds]
    report = {"model": "ConvNeXt-Tiny + Swin V2 feature fusion", "folds": results}
    if len(results) == 5:
        arrays = [np.load(args.output_root / f"fold_{fold}_validation_predictions.npz") for fold in args.folds]
        report["pooled_out_of_fold_metrics"] = compute_metrics(
            np.concatenate([array["label"] for array in arrays]),
            np.concatenate([array["probability"] for array in arrays]),
        )
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
