# ResNet-50 five-fold benchmark

Mean ± sample standard deviation (ddof=1) across five best checkpoints.

| Metric | Mean ± std | Paper Table 2 | Difference |
|---|---:|---:|---:|
| validation_loss | 0.635003 ± 0.008014 | Not reported | — |
| accuracy | 0.794091 ± 0.013427 | 0.7180 | +0.0761 |
| macro_f1 | 0.655405 ± 0.025108 | Not reported | — |
| macro_auc | 0.925465 ± 0.005376 | 0.8430 | +0.0825 |
| cohens_kappa_unweighted | 0.695839 ± 0.020111 | 0.7940 | -0.0982 |
| mcc | 0.699985 ± 0.020328 | 0.5560 | +0.1440 |

| Fold | Best epoch | Stopping epoch | Checkpoint |
|---|---:|---:|---|
| 0 | 10 | 15 | /workspace/project/training/runs/resnet50/fold_0/best.pt |
| 1 | 6 | 11 | /workspace/project/training/runs/resnet50/fold_1/best.pt |
| 2 | 4 | 9 | /workspace/project/training/runs/resnet50/fold_2/best.pt |
| 3 | 5 | 10 | /workspace/project/training/runs/resnet50/fold_3/best.pt |
| 4 | 4 | 9 | /workspace/project/training/runs/resnet50/fold_4/best.pt |

Paper Table 2 does not report ResNet-50 macro F1 or validation loss.
The reported paper kappa of 0.794 exceeds its accuracy of 0.718; this cannot be ordinary unweighted kappa on the same predictions. Weighting or a reporting error requires author clarification.
This benchmark uses unweighted kappa and preserves the established image-level splits and documented reproduction assumptions.
