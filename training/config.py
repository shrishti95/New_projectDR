"""Paper parameters and explicit reproduction assumptions for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import warnings


PAPER_BACKBONES = (
    "resnet34",
    "resnet50",
    "densenet121",
    "efficientnet_b7",
    "convnext_tiny",
    "vit_b_16",
)

UNDERSPECIFIED_TRAINING_ASSUMPTIONS = {
    "split_level": "image-level because APTOS provides no patient identifier",
    "random_seed": 42,
    "adamw_weight_decay": 1e-4,
    "cosine_minimum_learning_rate": 1e-6,
    "class_weight_formula": "N / (number_of_classes * class_count)",
    "loss_interpretation": (
        "inverse-frequency weighted focal cross-entropy with gamma=2 and "
        "label smoothing=0.1"
    ),
    "classification_head": "replace native classifier with one linear 5-class layer",
    "early_stopping_tie_rule": "strictly lower validation loss",
    "data_loader_workers": 4,
}


@dataclass(frozen=True)
class TrainingConfig:
    num_classes: int = 5
    folds: int = 5
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    early_stopping_patience: int = 5
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    seed: int = 42
    weight_decay: float = 1e-4
    cosine_min_lr: float = 1e-6
    num_workers: int = 4

    def warn_about_assumptions(self) -> None:
        details = "; ".join(
            f"{name}={value}"
            for name, value in UNDERSPECIFIED_TRAINING_ASSUMPTIONS.items()
        )
        warnings.warn(
            "RobustDRNet leaves benchmark details underspecified. "
            f"Reproduction assumptions in use: {details}",
            UserWarning,
            stacklevel=2,
        )
