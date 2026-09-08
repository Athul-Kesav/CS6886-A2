# MobileNetV2 CIFAR-10 Quantization

The compression method is implemented quantization: tensors are scaled, rounded, clipped to a signed integer range, dequantized for the forward pass, and wrapped with a straight-through estimator. No quantization or compression API is used.

## Results

The selected operating point is 8-bit weights and 8-bit activations:

| Metric | Result |
|---|---:|
| Baseline top-1 accuracy | 93.05% |
| Quantized top-1 accuracy | 91.33--91.35% |
| Weight compression ratio | 4.00x |
| Activation compression ratio | 4.00x |
| Estimated compressed model size | 2.117 MB |
| Estimated full-model compression ratio | 3.9996x |

The size is analytical, not the physical size of the float32 PyTorch state dictionary. It uses `parameter_count * bit_width / 8` plus one float32 scale per quantized Conv2d/Linear layer. Activation size is estimated from hooked output element counts.

## Layout

- `trainer.py`: MobileNetV2 baseline training and checkpoint generation.
- `load_dataset.py`: CIFAR-10 loaders, augmentation, and normalization.
- `quantizer.py`: manual fake quantization, STE, Conv-BN folding, and layer replacement.
- `metrics.py`: analytical model-size accounting and hooked activation accounting.
- `eval.py`: checkpoint loading and top-1 evaluation.
- `sweep.py`: 25-point WandB grid over weight and activation bit widths.
- `plot_gen.py`: learning curves and confusion matrix generation.
- `baseline_learning_curves.png`, `confusion_matrix.png`, `wandbPlot.png`: report figures.

## Environment

The supplied environment was Python 3.11 with CUDA-enabled PyTorch. The recorded dependency snapshot is in the repository-level `requirements.txt`. A virtual environment is recommended.

```bash
cd Quantization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For a CUDA installation, use a PyTorch wheel compatible with the target driver. The exact local wheel used for the reported runs was `torch==2.6.0+cu124` with `torchvision==0.21.0+cu124`; the other packages are recorded in `requirements.txt`.

## Reproduce the baseline

Run from `new/` so relative dataset, checkpoint, CSV, and figure paths resolve correctly.

```bash
cd new
python ../new/trainer.py
python plot_gen.py
```

Training uses MobileNetV2 width multiplier 1.0, a stride-1 first convolution, CIFAR-10 random crop and horizontal flip, ImageNet normalization, AdamW with learning rate `1e-3`, weight decay `1e-4`, cosine annealing for 15 epochs, and batch size 32. The trainer writes `training_metrics.csv`, `test_predictions.npz`, and `model/mobilenetv2_cifar10_baseline.pth`.

## Reproduce the sweep

The sweep requires a WandB account and login.

```bash
cd new
wandb login
rm -f sweep_results.csv
python sweep.py
```

The grid is

```text
weight bits:     8, 6, 4, 3, 2
activation bits: 8, 6, 4, 3, 2
```

The sweep writes `sweep_results.csv` and logs the same metrics to the WandB project `CS6886-Quantization-Sweep`. It evaluates the baseline checkpoint; it does not retrain for each bit width.

## Report figures

```bash
cd new
python plot_gen.py
```

This expects `training_metrics.csv` and `test_predictions.npz`, and writes `baseline_learning_curves.png` and `confusion_matrix.png`. `wandbPlot.png` is the exported parallel-coordinates figure.

## Reproducibility notes

The trainer seeds Python, NumPy, and PyTorch with seed 51 and enables deterministic cuDNN settings. `load_dataset.py` also calls `torch.manual_seed(17)`, and exact bitwise reproducibility can still vary with GPU, CUDA, cuDNN, and worker scheduling. BatchNorm is folded only after switching the copied evaluation model to `eval()`.

The current results directory contains historical sweep rows from two CSV schemas. Delete `sweep_results.csv` before a clean rerun; do not append new eight-column rows to an old seven-column file.
