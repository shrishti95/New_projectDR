# ConvNeXt V2 Base — APTOS 2019 five-fold report

Status: 5/5 folds completed.
Primary checkpoint: lowest validation loss. Macro-F1 and QWK checkpoints are supplementary only.

Pretrained checkpoint: [timm/convnextv2_base.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in1k/resolve/7b29800e499fdc06de5b612970f3384dc8d29ca5/model.safetensors)
Revision: `7b29800e499fdc06de5b612970f3384dc8d29ca5`; SHA-256: `ec152f1e375edc2b3dfac7a81155a449b4c5cbb7c5cf0b9494838f6c87518d73`.
Feature dimension: 1024; total parameters: 87,697,925; head-stage trainable: 5,125; partial-stage trainable: 85,518,853.

| Metric | ConvNeXt V2 Base mean ± sample std | Reproduced ConvNeXt-Tiny | Paper ConvNeXt-Tiny |
|---|---:|---:|---:|
| accuracy | 0.801740 ± 0.009490 | 0.806660 ± 0.021794 | 0.821 |
| macro_f1 | 0.678085 ± 0.016043 | 0.680059 ± 0.026837 | Not reported |
| macro_auc | 0.932572 ± 0.005709 | 0.933645 ± 0.009329 | 0.927 |
| cohens_kappa_unweighted | 0.708721 ± 0.014126 | 0.716978 ± 0.029172 | 0.727 |
| qwk | 0.885044 ± 0.013891 | 0.887876 ± 0.013519 | Not reported |
| mcc | 0.713361 ± 0.014210 | 0.723937 ± 0.024981 | 0.728 |

## Fold 0
Best epoch: 10; stopping epoch: 15; final stage: partial; training seconds: 1193.6.

| Metric | Official checkpoint |
|---|---:|
| accuracy | 0.807640 |
| macro_f1 | 0.686323 |
| macro_auc | 0.925564 |
| cohens_kappa_unweighted | 0.716713 |
| qwk | 0.890480 |
| mcc | 0.720741 |
| ece | 0.307845 |
| validation_loss | 0.624485 |

| Grade | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.988732 | 0.972299 | 0.980447 | 361 |
| 1 | 0.531532 | 0.797297 | 0.637838 | 74 |
| 2 | 0.802632 | 0.610000 | 0.693182 | 200 |
| 3 | 0.454545 | 0.641026 | 0.531915 | 39 |
| 4 | 0.583333 | 0.593220 | 0.588235 | 59 |

Confusion matrix (rows=true grade; columns=predicted grade):

| True / predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 351 | 8 | 1 | 0 | 1 |
| 1 | 3 | 59 | 11 | 0 | 1 |
| 2 | 1 | 41 | 122 | 20 | 16 |
| 3 | 0 | 1 | 6 | 25 | 7 |
| 4 | 0 | 2 | 12 | 10 | 35 |

Grade 1 recall: 0.797297; Grade 3 recall: 0.641026. Grade 2→3: 20; Grade 3→2: 6.

Supplementary selection (not substituted for primary results):

| Selection | Epoch | Accuracy | Macro F1 | QWK |
|---|---:|---:|---:|---:|
| macro_f1 | 10 | 0.807640 | 0.686323 | 0.890480 |
| qwk | 15 | 0.807640 | 0.675302 | 0.890819 |

## Fold 1
Best epoch: 12; stopping epoch: 17; final stage: partial; training seconds: 1385.0.

| Metric | Official checkpoint |
|---|---:|
| accuracy | 0.799454 |
| macro_f1 | 0.687174 |
| macro_auc | 0.932986 |
| cohens_kappa_unweighted | 0.706269 |
| qwk | 0.897150 |
| mcc | 0.713381 |
| ece | 0.291589 |
| validation_loss | 0.621945 |

| Grade | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.980609 | 0.980609 | 0.980609 | 361 |
| 1 | 0.534653 | 0.729730 | 0.617143 | 74 |
| 2 | 0.821705 | 0.530000 | 0.644377 | 200 |
| 3 | 0.394737 | 0.769231 | 0.521739 | 39 |
| 4 | 0.636364 | 0.711864 | 0.672000 | 59 |

Confusion matrix (rows=true grade; columns=predicted grade):

| True / predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 354 | 7 | 0 | 0 | 0 |
| 1 | 7 | 54 | 10 | 0 | 3 |
| 2 | 0 | 37 | 106 | 39 | 18 |
| 3 | 0 | 1 | 5 | 30 | 3 |
| 4 | 0 | 2 | 8 | 7 | 42 |

Grade 1 recall: 0.729730; Grade 3 recall: 0.769231. Grade 2→3: 39; Grade 3→2: 5.

Supplementary selection (not substituted for primary results):

| Selection | Epoch | Accuracy | Macro F1 | QWK |
|---|---:|---:|---:|---:|
| macro_f1 | 17 | 0.832196 | 0.717676 | 0.914936 |
| qwk | 17 | 0.832196 | 0.717676 | 0.914936 |

## Fold 2
Best epoch: 8; stopping epoch: 13; final stage: partial; training seconds: 1004.3.

| Metric | Official checkpoint |
|---|---:|
| accuracy | 0.813097 |
| macro_f1 | 0.693563 |
| macro_auc | 0.930379 |
| cohens_kappa_unweighted | 0.725966 |
| qwk | 0.882945 |
| mcc | 0.729861 |
| ece | 0.325177 |
| validation_loss | 0.618228 |

| Grade | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.994186 | 0.947368 | 0.970213 | 361 |
| 1 | 0.530973 | 0.810811 | 0.641711 | 74 |
| 2 | 0.824242 | 0.680000 | 0.745205 | 200 |
| 3 | 0.460317 | 0.743590 | 0.568627 | 39 |
| 4 | 0.604167 | 0.491525 | 0.542056 | 59 |

Confusion matrix (rows=true grade; columns=predicted grade):

| True / predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 342 | 19 | 0 | 0 | 0 |
| 1 | 0 | 60 | 13 | 0 | 1 |
| 2 | 1 | 28 | 136 | 20 | 15 |
| 3 | 0 | 0 | 7 | 29 | 3 |
| 4 | 1 | 6 | 9 | 14 | 29 |

Grade 1 recall: 0.810811; Grade 3 recall: 0.743590. Grade 2→3: 20; Grade 3→2: 7.

Supplementary selection (not substituted for primary results):

| Selection | Epoch | Accuracy | Macro F1 | QWK |
|---|---:|---:|---:|---:|
| macro_f1 | 9 | 0.830832 | 0.713548 | 0.893183 |
| qwk | 10 | 0.822647 | 0.695552 | 0.895114 |

## Fold 3
Best epoch: 9; stopping epoch: 14; final stage: partial; training seconds: 1116.2.

| Metric | Official checkpoint |
|---|---:|
| accuracy | 0.800546 |
| macro_f1 | 0.668806 |
| macro_auc | 0.941306 |
| cohens_kappa_unweighted | 0.706587 |
| qwk | 0.892687 |
| mcc | 0.711295 |
| ece | 0.305460 |
| validation_loss | 0.597815 |

| Grade | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.991573 | 0.977839 | 0.984658 | 361 |
| 1 | 0.549020 | 0.756757 | 0.636364 | 74 |
| 2 | 0.804054 | 0.595000 | 0.683908 | 200 |
| 3 | 0.361111 | 0.684211 | 0.472727 | 38 |
| 4 | 0.592593 | 0.542373 | 0.566372 | 59 |

Confusion matrix (rows=true grade; columns=predicted grade):

| True / predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 353 | 7 | 0 | 1 | 0 |
| 1 | 3 | 56 | 12 | 3 | 0 |
| 2 | 0 | 36 | 119 | 30 | 15 |
| 3 | 0 | 0 | 5 | 26 | 7 |
| 4 | 0 | 3 | 12 | 12 | 32 |

Grade 1 recall: 0.756757; Grade 3 recall: 0.684211. Grade 2→3: 30; Grade 3→2: 5.

Supplementary selection (not substituted for primary results):

| Selection | Epoch | Accuracy | Macro F1 | QWK |
|---|---:|---:|---:|---:|
| macro_f1 | 11 | 0.836066 | 0.701110 | 0.902791 |
| qwk | 8 | 0.818306 | 0.684185 | 0.903057 |

## Fold 4
Best epoch: 13; stopping epoch: 18; final stage: partial; training seconds: 1306.1.

| Metric | Official checkpoint |
|---|---:|
| accuracy | 0.787962 |
| macro_f1 | 0.654560 |
| macro_auc | 0.932627 |
| cohens_kappa_unweighted | 0.688071 |
| qwk | 0.861958 |
| mcc | 0.691529 |
| ece | 0.285233 |
| validation_loss | 0.631713 |

| Grade | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.991453 | 0.963989 | 0.977528 | 361 |
| 1 | 0.524752 | 0.716216 | 0.605714 | 74 |
| 2 | 0.764706 | 0.587940 | 0.664773 | 199 |
| 3 | 0.444444 | 0.526316 | 0.481928 | 38 |
| 4 | 0.469136 | 0.644068 | 0.542857 | 59 |

Confusion matrix (rows=true grade; columns=predicted grade):

| True / predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 348 | 12 | 0 | 0 | 1 |
| 1 | 2 | 53 | 15 | 2 | 2 |
| 2 | 1 | 31 | 117 | 19 | 31 |
| 3 | 0 | 1 | 8 | 20 | 9 |
| 4 | 0 | 4 | 13 | 4 | 38 |

Grade 1 recall: 0.716216; Grade 3 recall: 0.526316. Grade 2→3: 19; Grade 3→2: 8.

Supplementary selection (not substituted for primary results):

| Selection | Epoch | Accuracy | Macro F1 | QWK |
|---|---:|---:|---:|---:|
| macro_f1 | 16 | 0.824897 | 0.697492 | 0.888726 |
| qwk | 16 | 0.824897 | 0.697492 | 0.888726 |

Comparisons include differences in architecture, fine-tuning and augmentation; not a controlled architecture-only ablation.
Paper Table 2 does not report macro F1 or QWK; paper kappa weighting is unspecified.
Validation-selected checkpoints; no untouched external-test estimate.
Train exports are in-sample features, not OOF stacking features.

Exact checkpoint provenance: `pretrained_source.json`. Fixed experiment: `protocol.json`.
