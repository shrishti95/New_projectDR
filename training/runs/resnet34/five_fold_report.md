# ResNet-34 five-fold benchmark

Mean ± sample standard deviation (ddof=1) across five best checkpoints.

| Metric | Mean ± std | Paper Table 2 | Difference |
|---|---:|---:|---:|
| validation_loss | 0.649354 ± 0.017295 | Not reported | — |
| accuracy | 0.777172 ± 0.023650 | 0.7900 | -0.0128 |
| macro_f1 | 0.637618 ± 0.033254 | Not reported | — |
| macro_auc | 0.922166 ± 0.008515 | 0.9080 | +0.0142 |
| cohens_kappa_unweighted | 0.674254 ± 0.032195 | 0.6780 | -0.0037 |
| mcc | 0.681756 ± 0.029043 | 0.6790 | +0.0028 |

| Fold | Best epoch | Stopping epoch | Checkpoint |
|---|---:|---:|---|
| 0 | 4 | 9 | /workspace/project/training/runs/resnet34/fold_0/best.pt |
| 1 | 6 | 11 | /workspace/project/training/runs/resnet34/fold_1/best.pt |
| 2 | 4 | 9 | /workspace/project/training/runs/resnet34/fold_2/best.pt |
| 3 | 3 | 8 | /workspace/project/training/runs/resnet34/fold_3/best.pt |
| 4 | 7 | 12 | /workspace/project/training/runs/resnet34/fold_4/best.pt |

Paper does not report ResNet-34 macro F1 or validation loss in Table 2.
Paper does not identify kappa weighting; this run uses unweighted kappa.
Existing image-level split and reproduction assumptions are preserved.
