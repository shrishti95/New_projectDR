"""Train one paper benchmark backbone on one held-out APTOS fold."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info

from preprocessing import AptosZipDataset
from preprocessing.dataset import AptosRecord
from training.config import TrainingConfig
from training.losses import WeightedFocalCrossEntropy
from training.metrics import classification_metrics
from training.models import SUPPORTED_MODELS, build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker = get_worker_info()
    if worker is None:
        return
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    augmentation = getattr(worker.dataset, "augmentation", None)
    if augmentation is not None:
        augmentation.set_random_seed(seed)


def load_fold_records(
    manifest_path: str | Path,
    validation_fold: int,
) -> tuple[list[AptosRecord], list[AptosRecord]]:
    train_records: list[AptosRecord] = []
    validation_records: list[AptosRecord] = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record = AptosRecord(
                image_id=row["image_id"],
                diagnosis=int(row["diagnosis"]),
            )
            destination = (
                validation_records
                if int(row["fold"]) == validation_fold
                else train_records
            )
            destination.append(record)
    return train_records, validation_records


def inverse_frequency_weights(
    records: list[AptosRecord],
    classes: int,
) -> torch.Tensor:
    counts = np.bincount(
        [int(record.diagnosis) for record in records],
        minlength=classes,
    )
    if np.any(counts == 0):
        raise ValueError(f"Training fold has an empty class: {counts.tolist()}")
    weights = len(records) / (classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        samples = labels.shape[0]
        total_loss += float(loss.detach()) * samples
        total_samples += samples
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return total_loss / total_samples


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, list[str]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_image_ids: list[str] = []
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)
        probabilities = torch.softmax(logits.float(), dim=1)
        samples = labels.shape[0]
        total_loss += float(loss) * samples
        total_samples += samples
        all_targets.append(labels.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
        all_image_ids.extend(batch["image_id"])
        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    metrics = classification_metrics(targets, probabilities)
    return (
        total_loss / total_samples,
        metrics,
        targets,
        probabilities,
        all_image_ids,
    )


def make_loader(
    dataset: AptosZipDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--zip",
        default="/workspace/aptos2019-blindness-detection.zip",
    )
    parser.add_argument(
        "--manifest",
        default="/workspace/project/training/splits/aptos_five_folds.csv",
    )
    parser.add_argument(
        "--output-root",
        default="/workspace/project/training/runs",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )
    config.warn_about_assumptions()
    seed_everything(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for benchmark training")
    device = torch.device("cuda")
    train_records, validation_records = load_fold_records(
        args.manifest,
        args.fold,
    )
    train_dataset = AptosZipDataset(
        args.zip,
        train_records,
        training=True,
        warn_about_assumptions=False,
    )
    validation_dataset = AptosZipDataset(
        args.zip,
        validation_records,
        training=False,
        warn_about_assumptions=False,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        workers=config.num_workers,
        seed=config.seed + args.fold,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        workers=config.num_workers,
        seed=config.seed + 1000 + args.fold,
    )

    model = build_model(
        args.model,
        num_classes=config.num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)
    class_weights = inverse_frequency_weights(
        train_records,
        config.num_classes,
    ).to(device)
    criterion = WeightedFocalCrossEntropy(
        class_weights,
        gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.cosine_min_lr,
    )

    run_directory = (
        Path(args.output_root) / args.model / f"fold_{args.fold}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "best.pt"
    history_path = run_directory / "history.jsonl"
    metadata = {
        "model": args.model,
        "fold": args.fold,
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "class_weights": class_weights.detach().cpu().tolist(),
        "device": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "config": config.__dict__,
        "pretrained": not args.no_pretrained,
        "amp_bfloat16": args.amp,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }
    (run_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        started = time.monotonic()
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp=args.amp,
            max_batches=args.max_train_batches,
        )
        validation_loss, metrics, _, _, _ = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            amp=args.amp,
            max_batches=args.max_val_batches,
        )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.monotonic() - started,
            **metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "validation_loss": validation_loss,
                    "metrics": metrics,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print("early_stopping=True", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    final_loss, final_metrics, targets, probabilities, image_ids = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        amp=args.amp,
        max_batches=args.max_val_batches,
    )
    np.savez_compressed(
        run_directory / "validation_predictions.npz",
        image_id=np.asarray(image_ids),
        target=targets,
        probability=probabilities,
    )
    summary = {
        "best_epoch": checkpoint["epoch"],
        "validation_loss": final_loss,
        **final_metrics,
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
