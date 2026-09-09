# EfficientNet-B7 five-fold benchmark

Mean ± sample standard deviation (ddof=1) across five best checkpoints.

| Metric | Mean ± std | Paper |
|---|---:|---:|
| validation_loss | 0.643727 ± 0.009635 | Not reported |
| accuracy | 0.784550 ± 0.029337 | 0.792 |
| macro_f1 | 0.648068 ± 0.028366 | Not reported |
| macro_auc | 0.925786 ± 0.009412 | 0.898 |
| cohens_kappa_unweighted | 0.684880 ± 0.038804 | 0.829 |
| mcc | 0.691663 ± 0.033480 | 0.672 |

Paper Table 2 does not report EfficientNet-B7 macro F1 or validation loss.
Paper kappa 0.829 exceeds accuracy 0.792 and cannot be ordinary unweighted kappa on the same predictions. Weighting or reporting needs clarification.
Unweighted kappa, the existing image-level splits and documented reproduction assumptions are preserved.
