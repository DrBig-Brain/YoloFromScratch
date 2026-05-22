# YoloFromScratch

[![Status](https://img.shields.io/badge/status-experimental-yellow.svg)](https://github.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A compact, educational reimplementation of YOLOv1 in PyTorch — written for learning, experimentation, and small datasets. This repo includes a lightweight model, training loop, utilities for converting and evaluating predictions, and helper scripts for visualization and checkpointing.

🚀 Features
- Minimal YOLOv1-inspired architecture in `model.py`
- Training driver in `train.py` with checkpointing
- Utilities for bbox conversion, NMS, IoU, and mAP in `utils/`
- Tiny example dataset in `data/` for quick iterations

💡 Goal
This project is not meant to match production YOLO performance. It exists to illustrate core YOLO ideas (cell-based predictions, objectness, coordinate loss, NMS, mAP) with readable code you can step through and modify.

Table of contents
- [YoloFromScratch](#yolofromscratch)
  - [Quickstart](#quickstart)
  - [Repository structure](#repository-structure)
  - [Training](#training)
  - [Evaluation \& Visualization](#evaluation--visualization)
  - [Tips for better results](#tips-for-better-results)
  - [Contributing](#contributing)
  - [License](#license)

---

## Quickstart
1. Create a Python environment (recommended) and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
# add packages you need, example:
pip install torch torchvision tqdm matplotlib
```

2. Inspect the small example dataset in `data/` and `data/labels/`.
3. Train the model (quick test / debugging):

```bash
python train.py
```

Note: Hyperparameters are set at top of `train.py` (batch size, learning rate, epochs). For quick runs set `EPOCHS` low and `BATCH_SIZE` small.

## Repository structure
- `train.py` — training loop, data loaders, mAP calculation and checkpointing
- `model.py` — compact YOLOv1-like model
- `dataset.py` — VOC-style dataset loader for CSV annotations
- `loss.py` — YOLO loss implementation
- `utils/` — helper modules
  - `cellboxes_to_boxes.py`, `convert_cellboxes.py`, `get_bboxes.py`
  - `non_max_supression.py`, `intersection_over_union.py`
  - `mean_average_precision.py`, `plot_image.py`, `save_checkpoint.py`, `load_checkpoint.py`
- `data/` — sample CSVs and label files plus `images/` directory

## Training
- Edit hyperparameters at the top of `train.py`.
- To resume from a checkpoint, set `LOAD_MODEL=True` and point `LOAD_MODEL_FILE` to the checkpoint file.
- Checkpoints are saved via `utils/save_checkpoint.py` as `overfit.pth.tar` by default.

Example quick-train command for CPU-only machine (reduce workers):

```bash
python train.py
```

## Evaluation & Visualization
- The repo computes mAP using `utils/mean_average_precision.py`.
- To visualize predictions manually, use `utils/plot_image.py` together with `CellboxesToBoxes` and `NonMaxSuppression` to render predicted boxes on images.

## Tips for better results
- Dataset size: YOLO needs substantial data. The included `data/8examples.csv` is for debugging.
- Hyperparameters: try higher learning rate, lower weight decay, sensible batch sizes.
- Augmentation: add random flips / color jitter in `transform` for robustness.
- Thresholds: confidence & NMS IoU thresholds affect detection recall/precision — tune these.
- Check model outputs: ensure objectness/confidence entries are in [0,1] and widths/heights are positive.

## Contributing
Happy to accept improvements — especially:
- Better, numerically-stable box parameterization
- More robust training recipes (LR schedules, augmentations)
- Cleaner dataset parsing and COCO-style evaluation support

Open an issue or send a PR with an explanation of the change.

## License
This repository is provided under the MIT license. See `LICENSE` for details.

---

Need a custom README style (fancier badges, GIF demo, CI status, or a longer tutorial)? I can add a live demo GIF and a one-page explanation of the loss and bbox math next. 🎉
