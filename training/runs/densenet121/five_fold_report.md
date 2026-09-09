# DenseNet-121 five-fold benchmark

Mean ± sample standard deviation (ddof=1) across five best checkpoints.

| Metric | Mean ± std | Paper Table 2 | Difference |
|---|---:|---:|---:|
| validation_loss | 0.649096 ± 0.025597 | Not reported | — |
| accuracy | 0.775806 ± 0.013173 | 0.7900 | -0.0142 |
| macro_f1 | 0.632319 ± 0.022742 | Not reported | — |
| macro_auc | 0.920137 ± 0.010883 | 0.9040 | +0.0161 |
| cohens_kappa_unweighted | 0.672686 ± 0.017728 | 0.8180 | -0.1453 |
| mcc | 0.679499 ± 0.017903 | 0.6530 | +0.0265 |

| Fold | Best epoch | Stopping epoch | Checkpoint |
|---|---:|---:|---|
| 0 | 5 | 10 | /workspace/project/training/runs/densenet121/fold_0/best.pt |
| 1 | 6 | 11 | /workspace/project/training/runs/densenet121/fold_1/best.pt |
| 2 | 4 | 9 | /workspace/project/training/runs/densenet121/fold_2/best.pt |
| 3 | 5 | 10 | /workspace/project/training/runs/densenet121/fold_3/best.pt |
| 4 | 4 | 9 | /workspace/project/training/runs/densenet121/fold_4/best.pt |

Paper Table 2 does not report DenseNet-121 macro F1 or validation loss.
Paper kappa 0.818 exceeds accuracy 0.790 and cannot be ordinary unweighted kappa on the same predictions. Weighting or a reporting error requires clarification.
The existing image-level splits and documented reproduction assumptions are preserved; this run reports unweighted kappa.
