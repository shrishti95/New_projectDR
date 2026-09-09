# Project backup

Target: https://github.com/shrishti95/New_projectDR

The Git repository includes the base paper, preprocessing and training code,
split manifests, available logs, reports, predictions and model checkpoints.
Checkpoints and prediction archives use Git LFS. A clone needs Git LFS installed
and access to the repository's LFS objects to retrieve those files.

Reinstallable virtual environments, download caches, duplicate ZIP exports,
temporary notebook files and credentials are excluded. The APTOS dataset is
outside this project at /workspace/aptos2019-blindness-detection.zip and is
not included. Live training can produce newer results after the backup snapshot.

Environment used: Python 3.12, torch 2.9.0+cu128, torchvision 0.24.0+cu128,
numpy 2.2.6, albumentations 2.0.8, opencv-python-headless 4.12.0.88.
Scripts currently use /workspace paths that may need adapting on another host.
