# RobustDRNet: completed benchmark comparison

Snapshot: 2026-09-09. Source: classification.pdf, page 8, Table 2.
This work reports the arithmetic mean ± sample standard deviation (ddof=1)
of five validation results, each evaluated after restoring its minimum
validation-loss checkpoint. Paper values are the single values reported in Table 2.

## Completed models

| Model | Metric | Paper | This work: mean ± std | Mean minus paper |
|---|---|---:|---:|---:|
| ResNet-34 | Accuracy | 79.00% | 77.72% ± 2.37 percentage points | −1.28 percentage points |
| ResNet-34 | Macro AUC | 0.9080 | 0.9222 ± 0.0085 | +0.0142 |
| ResNet-34 | Cohen's kappa | 0.6780 | 0.6743 ± 0.0322 | −0.0037 |
| ResNet-34 | MCC | 0.6790 | 0.6818 ± 0.0290 | +0.0028 |
| ResNet-34 | Macro F1 | Not reported | 0.6376 ± 0.0333 | — |
| ResNet-34 | Validation loss | Not reported | 0.6494 ± 0.0173 | — |
| ResNet-50 | Accuracy | 71.80% | 79.41% ± 1.34 percentage points | +7.61 percentage points |
| ResNet-50 | Macro AUC | 0.8430 | 0.9255 ± 0.0054 | +0.0825 |
| ResNet-50 | Cohen's kappa | 0.7940 | 0.6958 ± 0.0201 | −0.0982* |
| ResNet-50 | MCC | 0.5560 | 0.7000 ± 0.0203 | +0.1440 |
| ResNet-50 | Macro F1 | Not reported | 0.6554 ± 0.0251 | — |
| ResNet-50 | Validation loss | Not reported | 0.6350 ± 0.0080 | — |

*Kappa is unweighted in this work. The paper does not identify weighting.
Its ResNet-50 kappa of 0.794 exceeds accuracy 0.718, which cannot occur
for ordinary unweighted kappa on the same predictions. This may reflect
weighting or a reporting error; the displayed subtraction is not evidence
of a like-for-like performance difference.

## Interpretation

ResNet-34 has accuracy 1.28 percentage points below the published figure,
with a higher macro AUC and similar kappa and MCC. ResNet-50 has higher
reported accuracy, macro AUC, and MCC than the paper's ResNet-50 figures.
These are numerical comparisons, not evidence of statistically significant
superiority or an exact replication: the authors' splits, several parameters,
and the precise combined-loss formulation are unavailable or underspecified.
No significance test against the paper's unavailable per-image predictions
or per-fold results has been performed.

## Pending at this snapshot

DenseNet-121 is training fold 4. EfficientNet-B7 is waiting for DenseNet-121.
ConvNeXt-Tiny and ViT-B/16 have no completed five-fold results in this snapshot.
No partial model results are included in the comparison table.

## Result files

- resnet34/five_fold_report.json
- resnet34/five_fold_report.md
- resnet50/five_fold_report.json
- resnet50/five_fold_report.md

The JSON reports contain full-precision metrics, per-fold best/stopping
epochs, and checkpoint paths. Training settings and split assignments have
not been changed to produce this report.
