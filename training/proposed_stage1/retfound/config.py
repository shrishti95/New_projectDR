from dataclasses import dataclass, asdict
from pathlib import Path
from preprocessing.config import PreprocessingConfig
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
MANIFEST=ROOT/'training/splits/aptos_five_folds.csv'
MANIFEST_SHA256='49c00bc9bf5272330a727ae8d6a7e6e6474d48f68070dd913b644d7b4f78c1d1'
CHECKPOINT=HERE/'checkpoints/RETFound_mae_natureCFP.pth'
@dataclass(frozen=True)
class Config:
 seed:int=42; head_epochs:int=3; max_epochs:int=30; patience:int=5
 batch_size:int=32; effective_batch_size:int=32; workers:int=4
 head_warmup_lr:float=1e-3; backbone_lr:float=5e-6; head_lr:float=5e-5
 weight_decay:float=1e-4; gamma:float=2.; smoothing:float=.1; gradient_clip:float=1.; final_blocks:int=4
 bf16:bool=True

def preprocessing(): return PreprocessingConfig(augmentation_seed=42)
def protocol(): return {'model':'RETFound_mae_natureCFP / ViT-Large/16','pretrained_source':'https://huggingface.co/YukunZhou/RETFound_mae_natureCFP','pretraining':'1.6M retinal images; official RETFound Nature CFP checkpoint','manifest_sha256':MANIFEST_SHA256,'training':asdict(Config()),'stage1':'3 epochs head only, AdamW LR 1e-3','stage2':'final 4 transformer blocks plus final norm; backbone 5e-6, head 5e-5','primary_checkpoint':'best_loss.pt','validation_use':'minimum validation loss only'}
