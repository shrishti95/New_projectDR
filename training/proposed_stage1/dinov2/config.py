from dataclasses import dataclass, asdict
from pathlib import Path
from preprocessing.config import PreprocessingConfig

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODEL = "vit_base_patch14_dinov2.lvd142m"
MANIFEST = ROOT / "training/splits/aptos_five_folds.csv"
MANIFEST_SHA256 = "49c00bc9bf5272330a727ae8d6a7e6e6474d48f68070dd913b644d7b4f78c1d1"

@dataclass(frozen=True)
class Config:
    seed:int=42; head_epochs:int=3; max_epochs:int=30; patience:int=5
    batch_size:int=32; effective_batch_size:int=32; workers:int=4
    head_warmup_lr:float=1e-3; backbone_lr:float=1e-5; head_lr:float=1e-4
    weight_decay:float=1e-4; gamma:float=2.; smoothing:float=.1
    gradient_clip:float=1.; bf16:bool=True; final_blocks:int=2

def preprocessing():
    return PreprocessingConfig(augmentation_seed=42)

def protocol():
    return {"model":MODEL,"pretrained_source":"timm pretrained DINOv2 lvd142m",
            "manifest_sha256":MANIFEST_SHA256,"training":asdict(Config()),
            "stage1":"3 epochs head only, AdamW LR 1e-3",
            "stage2":"final 2 transformer blocks plus head; backbone 1e-5, head 1e-4",
            "primary_checkpoint":"best_loss.pt","validation_use":"selection and reporting only"}
