# Proposed stage 1: ConvNeXt V2 Base

This experiment preserves the reproduction's exact 3,662-image five-fold
manifest (SHA-256 in config.py); it does not regenerate splits. There is no
external test-set use, oversampling or sample removal. The primary result is
always the minimum-validation-loss checkpoint. F1/QWK selections are secondary.

## Predeclared training

- ImageNet-1k FCMAE + ImageNet-1k fine-tuned ConvNeXt V2 Base, `timm==1.0.19`.
  The exact revision, weights SHA-256 and download URL are in
  `pretrained_source.json`. A fresh five-class linear head is used in every fold.
- Resolution 224×224. Reuse resize → LAB CLAHE → circular mask → training-only
  augmentation → ImageNet normalization from the validated reproduction.
- Horizontal flip p=0.5; vertical flip p=0.1 (already present in reproduction);
  rotation ±15° p=0.5; scale 0.95–1.05 and translation ±3% p=0.3; brightness/
  contrast ±0.1 p=0.3. Coarse dropout is at most one 1–3%-side patch with p=0.05.
  Numerical augmentation settings are predeclared, not fitted to validation.
- Head-only epochs 1–3: freeze everything except final linear classifier;
  AdamW LR 1e-3. Partial stage epochs 4–30: unfreeze final two ConvNeXt stages
  and final normalization; backbone LR 1e-5 and classifier LR 1e-4. AdamW weight
  decay 1e-4. Cosine LR decay over the 27 possible partial-stage epochs, eta_min=0.
  Full unfreezing is disabled. No aggressive LR increases.
- Weighted focal loss: gamma 2, smoothing 0.1, existing reproduction formulation.
  Compute weights N/(5*n_class) exclusively from each fold's training records.
  Compute loss in FP32 with BF16 model forward/backward and gradient clipping 1.0.
- Batch 32. Only CUDA OOM in the training-only smoke test can reduce the physical
  batch to 16 or 8; sample-weighted gradient accumulation preserves effective 32.
  No validation metric is involved in that decision.
- Seed 42 for every model; training DataLoader seed 42+fold and validation seed
  1042+fold as in the reproduction. Deterministic PyTorch algorithms, CuBLAS
  workspace configuration and four NumPy-transport workers avoid /dev/shm errors.
- Strict global best validation loss across both stages; patience 5 is active
  in stage 2 with its counter reset at the transition. All three checkpoint
  criteria use strict improvement, retaining the earlier checkpoint on ties.

## Outputs

Each `runs/fold_N` saves metadata, epoch history, `best_loss.pt`,
`best_macro_f1.pt`, `best_qwk.pt`, and a full summary. History includes stages,
LR groups, actual/effective batch sizes, trainable parameter counts, GPU memory,
per-epoch time and all requested validation metrics. Phase snapshots are
observational records, not independently controlled ablation experiments.

After restoring `best_loss.pt`, deterministic `train_features.npz` and
`validation_features.npz` store `sample_id`, `label`, `fold`, `feature` (1024),
`probability` (5) and `predicted_class`. IDs/labels are checked against the
original split. Training features are in-sample and must not be mislabeled OOF.

Metrics include accuracy, macro F1/AUC, unweighted kappa, QWK, MCC, per-class
precision/recall/F1, confusion matrix (true rows, predicted columns), and ECE
with 15 fixed bins. Grade 1/3 recall and Grade 2↔3 errors are explicit fields.

`logs/fold_N.log` prints every epoch and all final fold metrics.
`logs/experiment.log` prints each fold's official metrics and final model
mean ± sample standard deviation. `status.json` records live state and errors.
The requested `convnextv2_five_fold_report.md` updates after each fold and prints
five-fold aggregates only after all five are complete. It compares with reproduced
and paper ConvNeXt-Tiny; it separates supplementary checkpoints explicitly.

## Commands

Run from the repository root with `.venv/bin/python`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m training.proposed_stage1.convnextv2.smoke
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m training.proposed_stage1.convnextv2.run
```

The runner requires a passing smoke result and matching tested source hashes,
records an immutable protocol, and refuses to overwrite incomplete fold outputs.
Validation images never participate in backward passes. Pretrained weights and
fresh optimizer state are reinitialized per fold. Checkpoint selection uses
validation, so these are validation estimates, not an untouched final test.
This experiment changes architecture, augmentation and fine-tuning together;
its comparison does not isolate the causal effect of architecture alone.
