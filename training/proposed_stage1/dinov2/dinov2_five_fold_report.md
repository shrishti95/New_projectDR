# DINOv2 ViT-B/14 five-fold report

Official validation-loss checkpoints; mean ± sample standard deviation (ddof=1).

| Metric | DINOv2 | ViT-B/16 reproduced | ConvNeXt-Tiny reproduced | ConvNeXt V2 Base |
|---|---:|---:|---:|---:|
| accuracy | 0.777416 ± 0.041679 | 0.772537 | 0.806660 | 0.801740 |
| macro_f1 | 0.649687 ± 0.040755 | 0.635900 | 0.680059 | 0.678085 |
| macro_auc | 0.919974 ± 0.006196 | 0.926861 | 0.933645 | 0.932572 |
| cohens_kappa_unweighted | 0.678263 ± 0.053193 | 0.667642 | 0.716978 | 0.708721 |
| qwk | 0.864048 ± 0.044380 | Not reported | 0.887876 | 0.885044 |
| mcc | 0.687022 ± 0.046023 | 0.674752 | 0.723936 | 0.713361 |
| grade_1_recall | 0.762162 ± 0.051988 | Not reported | Not reported | Not reported |
| grade_3_recall | 0.693657 ± 0.087247 | Not reported | Not reported | Not reported |

| Fold | Best epoch | Accuracy | Macro F1 | Macro AUC | Kappa | QWK | MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7 | 0.792633 | 0.656950 | 0.917110 | 0.696855 | 0.888551 | 0.702059 |
| 1 | 8 | 0.777626 | 0.659173 | 0.923712 | 0.677859 | 0.869683 | 0.685528 |
| 2 | 9 | 0.792633 | 0.667604 | 0.928271 | 0.699396 | 0.868139 | 0.707419 |
| 3 | 12 | 0.816940 | 0.685173 | 0.918590 | 0.728432 | 0.904568 | 0.730269 |
| 4 | 7 | 0.707250 | 0.579537 | 0.912185 | 0.588772 | 0.789298 | 0.609836 |


DINOv2 mean accuracy, macro F1, QWK and MCC are below the reproduced ConvNeXt-Tiny and ConvNeXt V2 Base means. Its macro AUC is also below both ConvNeXt baselines but above reproduced ViT-B/16. Variability is higher than ConvNeXt V2 Base for every listed classification metric.

Grade 1 mean recall: 0.762162 ± 0.051988.
Grade 3 mean recall: 0.693657 ± 0.087247.
