# RETFound five-fold report

Official model: RETFound MAE ViT-Large/16, `RETFound_mae_natureCFP.pth` from the official Hugging Face repository.

| Metric | Mean ± sample SD |
|---|---:|
| accuracy | 0.749330 ± 0.015119 |
| macro_f1 | 0.613645 ± 0.018649 |
| macro_auc | 0.911507 ± 0.007330 |
| cohens_kappa_unweighted | 0.635765 ± 0.019156 |
| qwk | 0.835624 ± 0.013626 |
| mcc | 0.643199 ± 0.017214 |
| grade_1_recall | 0.708108 ± 0.038932 |
| grade_3_recall | 0.564642 ± 0.059893 |

## Fold results

| Fold | Best epoch | Val loss | Accuracy | Macro F1 | QWK | Grade 1 recall | Grade 3 recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 0.692086 | 0.729877 | 0.583754 | 0.822874 | 0.756757 | 0.512821 |
| 1 | 27 | 0.693625 | 0.740791 | 0.624615 | 0.828611 | 0.675676 | 0.641026 |
| 2 | 30 | 0.695310 | 0.751705 | 0.617682 | 0.828918 | 0.689189 | 0.564103 |
| 3 | 30 | 0.673489 | 0.754098 | 0.609901 | 0.840622 | 0.743243 | 0.605263 |
| 4 | 25 | 0.699924 | 0.770178 | 0.632273 | 0.857093 | 0.675676 | 0.500000 |
