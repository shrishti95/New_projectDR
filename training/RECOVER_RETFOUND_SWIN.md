# Recover completed RETFound and Swin V2 experiments

Install Git LFS before cloning, then run `git lfs pull` in the cloned repository.
The recovery folders contain one official validation-loss-selected checkpoint per fold.
These preserve original tensor values and dtypes, but omit optimizer/scheduler state.
They support inference and feature extraction, not exact mid-training resumption.
SHA-256 checksums and tensor equality verification are recorded in each recovery manifest.

Each model's runs/fold_* directory contains history, metadata, official and available
supplementary metrics, and train/validation NPZ features including IDs, labels and probabilities.
Use training/splits/aptos_five_folds.csv for the original split assignments.
Swin uses 256 input; RETFound uses 224. Refer to each experiment's configuration.

To load recovery weights with PyTorch:

```python
checkpoint = torch.load(path, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model_state'], strict=True)
```

Construct the original architecture with five output classes before loading.
RETFound: RETFound_mae(num_classes=5, global_pool=True, img_size=224)
from training.proposed_stage1.retfound.retfound_model.
Swin: timm.create_model('swinv2_base_window8_256.ms_in1k', pretrained=False, num_classes=5).
The environment used timm 1.0.19 and PyTorch 2.9.0+cu128.
Swin's runner reuses the tracked ConvNeXt V2 experiment utilities.

The APTOS dataset is NOT included: retain a separate backup or obtain it from its original source.
The original downloadable pretrained checkpoints, caches, logs, optimizer states,
and supplementary model weights are excluded. Pretrained checkpoint paths in metadata
refer to the old workspace; trained recovery weights do not require those downloads.
The official RETFound pretrained source is YukunZhou/RETFound_mae_natureCFP on Hugging Face.
