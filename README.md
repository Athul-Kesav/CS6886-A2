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

This is a flat project directory. Run the Python scripts from the project root so
their relative paths resolve correctly.

```text
Ass2/
├── README.md
├── load_dataset.py             # CIFAR-10 loaders and preprocessing
├── trainer.py                  # Baseline training and checkpoint generation
├── quantizer.py                # Fake quantization, STE, and Conv-BN folding
├── metrics.py                  # Model-size and activation-footprint metrics
├── eval.py                     # Checkpoint loading and evaluation helpers
├── sweep.py                    # WandB grid over weight and activation bit widths
├── plot_gen.py                 # Learning curves and confusion matrix generation
├── training_metrics.csv        # Generated baseline training history
├── test_predictions.npz        # Generated baseline predictions and labels
├── sweep_results.csv           # Generated quantization sweep results
├── baseline_learning_curves.png
├── confusion_matrix.png
├── wandbPlot.png
└── model/                      # Created by trainer.py
	└── mobilenetv2_cifar10_baseline.pth
```

## Environment

The supplied environment uses Python 3.11 with CUDA-enabled PyTorch. The pinned
dependencies are listed in `requirements.txt`. A virtual environment is
recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements use `torch==2.4.1` and `torchvision==0.19.1` with CUDA 12.1
runtime dependencies (`nvidia-cuda-runtime-cu12==12.1.105`). If installing
PyTorch separately, use wheels compatible with this CUDA 12.1 environment and
the target NVIDIA driver.

## Reproduce the baseline

Run these commands from the `Ass2/` project root. The first run downloads
CIFAR-10 into `data/` if it is not already available.

```bash
python trainer.py
python plot_gen.py
```

Training uses MobileNetV2 width multiplier 1.0, a stride-1 first convolution,
CIFAR-10 random crop and horizontal flip, ImageNet normalization, AdamW with
learning rate `1e-3`, weight decay `1e-4`, cosine annealing for 15 epochs, and
batch size 32. The trainer writes `training_metrics.csv`,
`test_predictions.npz`, and `model/mobilenetv2_cifar10_baseline.pth`.

## Reproduce the sweep

The sweep requires a WandB account and login.

```bash
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
python plot_gen.py
```

This expects `training_metrics.csv` and `test_predictions.npz`, and writes
`baseline_learning_curves.png` and `confusion_matrix.png`. `wandbPlot.png` is
the exported parallel-coordinates figure.

## Reproducibility notes

The trainer seeds Python, NumPy, and PyTorch with seed 51 and enables deterministic cuDNN settings. `load_dataset.py` also calls `torch.manual_seed(17)`.
