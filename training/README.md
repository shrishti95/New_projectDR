# Stage 2: individual benchmark models

The next RobustDRNet stage after preprocessing is to fine-tune six
ImageNet-pretrained five-class classifiers independently:

- ResNet-34
- ResNet-50
- DenseNet-121
- EfficientNet-B7
- ConvNeXt-Tiny
- ViT-B/16

The paper specifies AdamW, initial learning rate 1e-4, cosine annealing,
label smoothing 0.1, focal gamma 2, batch size 32, up to 30 epochs,
early-stopping patience 5, inverse-frequency class weighting, and stratified
five-fold cross-validation.

## Reproducibility boundary

APTOS provides an image ID and diagnosis but no patient ID. The paper claims
patient-wise stratification later in its discussion, but that cannot be
reconstructed from the supplied public archive. The generated split is
therefore explicitly image-level.

The paper also does not give its random seed, AdamW weight decay, minimum
cosine learning rate, exact class-weight formula, exact way focal loss is
combined with label smoothing, or detailed classification head. Runnable
choices are centralized and warned about in config.py.

The implementation reports unweighted Cohen kappa by name because the paper
does not state whether its reported kappa is unweighted or quadratic-weighted.

## Commands

Run from /workspace/project with the project-local Python:

    PYTHONPATH=/workspace/project .venv/bin/python -m training.create_folds
    PYTHONPATH=/workspace/project .venv/bin/python -m training.smoke_test

Train one backbone and fold:

    PYTHONPATH=/workspace/project .venv/bin/python \
      -m training.train_benchmark --model resnet34 --fold 0

The runner saves the best checkpoint, epoch history, metadata, summary, and
held-out probabilities. Those out-of-fold probabilities will later be used
for leakage-free stacking.
