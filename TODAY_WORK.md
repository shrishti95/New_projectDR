# Completed work — 2026-09-14

The previously missing final ViT-B/16 fold is included under
`training/stage3_final/runs/vit_b_16/fold_4/`, together with its completed
five-fold report. The first four ViT folds were already backed up and are not
duplicated here. Its best checkpoint is epoch 9; training stopped at epoch 14.

ConvNeXt V2 Base was trained and evaluated on the exact five APTOS folds used
in the reproduction. All five folds, official checkpoints and feature exports
were verified. The experiment follows the predeclared validation-loss selection
rule; macro-F1 and QWK checkpoints are supplementary.

Main report: [ConvNeXt V2 five-fold report](training/proposed_stage1/convnextv2/convnextv2_five_fold_report.md).

- Experiment source, fixed protocol, smoke-test evidence and environment manifest
  are in `training/proposed_stage1/convnextv2/`.
- `runs/fold_0` through `runs/fold_4` contain all three checkpoints, epoch histories,
  metadata, summaries, and deterministic train/validation feature archives.
- `logs/` contains the complete fold and experiment console logs.
- Checkpoints (`.pt`) and feature archives (`.npz`) are stored in Git LFS.
- The APTOS dataset, pretrained download cache, virtual environment and local
  credentials are not included. Download the dataset separately.

Primary mean ± sample standard deviation: accuracy 80.17% ± 0.95 percentage
points, macro F1 0.6781 ± 0.0160, macro AUC 0.9326 ± 0.0057, unweighted kappa
0.7087 ± 0.0141, QWK 0.8850 ± 0.0139, and MCC 0.7134 ± 0.0142. This configuration
did not improve mean metrics over the reproduced ConvNeXt-Tiny baseline.

Install Git LFS before cloning/downloading model artifacts. To fetch only this
experiment's artifacts from a clone:

```bash
git lfs pull --include='training/proposed_stage1/convnextv2/runs/**'
```

The provenance JSON preserves the original execution paths. On another machine,
restore the recorded pretrained checkpoint to the local cache and update its
local `path` in `pretrained_source.json`; keep the recorded revision and SHA-256.
The original reproduction's source and results are preserved. Only the required portable-loader and ViT-report support code is included.
Fusion work, prior interrupted-run directories and raw datasets remain excluded.
