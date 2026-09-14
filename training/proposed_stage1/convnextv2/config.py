from dataclasses import asdict, dataclass
from pathlib import Path
from preprocessing.config import PreprocessingConfig

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MANIFEST = ROOT / 'training/splits/aptos_five_folds.csv'
MANIFEST_SHA256 = '49c00bc9bf5272330a727ae8d6a7e6e6474d48f68070dd913b644d7b4f78c1d1'
MODEL = 'convnextv2_base.fcmae_ft_in1k'
HF_REPO = 'timm/' + MODEL


@dataclass(frozen=True)
class Config:
    seed: int = 42
    input_size: int = 224
    classes: int = 5
    head_epochs: int = 3
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 32
    effective_batch_size: int = 32
    workers: int = 4
    head_warmup_lr: float = 1e-3
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    gamma: float = 2.0
    smoothing: float = 0.1
    gradient_clip: float = 1.0
    bf16: bool = True
    full_unfreeze: bool = False
    cosine_min_lr: float = 0.0


def preprocessing():
    return PreprocessingConfig(
        horizontal_flip_probability=0.5, vertical_flip_probability=0.1,
        rotation_probability=0.5, rotation_limit_degrees=15.0,
        affine_probability=0.3, affine_scale_range=(0.95,1.05),
        affine_translation_fraction=(-0.03,0.03),
        brightness_contrast_probability=0.3, brightness_limit=0.1, contrast_limit=0.1,
        coarse_dropout_probability=0.05, coarse_dropout_holes_range=(1,1),
        coarse_dropout_size_fraction=(0.01,0.03), augmentation_seed=42,
    )


def protocol():
    return {'training':asdict(Config()), 'preprocessing':asdict(preprocessing()),
            'model':MODEL, 'manifest_sha256':MANIFEST_SHA256,
            'primary_checkpoint':'best_loss.pt',
            'supplementary_checkpoints':['best_macro_f1.pt','best_qwk.pt'],
            'stage1':'epochs 1-3: only final linear classifier',
            'stage2':'epochs 4-30: final two ConvNeXt stages, final normalization, classifier',
            'early_stopping':'strict validation-loss improvement; reset patience at stage 2 boundary; global best retained',
            'seed_policy':'seed 42 for every model; training loader 42+fold; validation loader 1042+fold',
            'full_unfreeze':'disabled before observing validation results',
            'class_weights':'N_train/(5*training_class_count)',
            'loss':'existing weighted focal cross-entropy with smoothed targets, computed in FP32',
            'ece':'15 fixed equal-width confidence bins; no calibration fitted',
            'feature_export':'deterministic training=False preprocessing for both splits; official best-loss model',
            'validation_use':'early stopping, checkpoint selection, reporting only',
            'stage_logs':'phase snapshots are observational, not controlled independent ablations'}
