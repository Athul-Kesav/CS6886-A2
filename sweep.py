import copy
import csv
import os

import torch
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from eval import evaluate_model, get_base_model
from quantizer import apply_quantization

from metrics import calculate_compression_metrics, measure_activation_footprint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "./model/mobilenetv2_cifar10_baseline.pth"
CSV_FILE = "sweep_results.csv"
CSV_COLUMNS = [
    "w_bits",
    "a_bits",
    "acc",
    "comp_ratio",
    "size_mb",
    "weight_cr",
    "act_cr",
    "act_cr_measured",
]


def make_testloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)


def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as file:
            csv.writer(file).writerow(CSV_COLUMNS)

def run_sweep():
    wandb.init()
    config = wandb.config
    testloader = make_testloader()
    ensure_csv()

    base_model = get_base_model(WEIGHTS_PATH, DEVICE)
    quant_model = copy.deepcopy(base_model)
    quant_model.eval()
    apply_quantization(quant_model, config.w_bits, config.a_bits)
    quant_model.to(DEVICE)

    acc = evaluate_model(quant_model, testloader, DEVICE)
    metrics = calculate_compression_metrics(quant_model, config.w_bits, config.a_bits)
    act_measured = measure_activation_footprint(
        quant_model, testloader, DEVICE, config.a_bits, num_batches=1
    )

    wandb.log({
        "weight_quant_bits": config.w_bits,
        "activation_quant_bits": config.a_bits,
        "quantized_acc": acc,
        "compression_ratio": metrics["compression_ratio"],
        "model_size_mb": metrics["compressed_mb"],
        "weight_cr": metrics["weight_cr"],
        "act_cr": metrics["act_cr"],
        "act_cr_measured": act_measured["act_cr_measured"],
        "activation_fp32_mb": act_measured["activation_fp32_mb"],
        "activation_quant_mb": act_measured["activation_quant_mb"],
    })

    with open(CSV_FILE, "a", newline="") as file:
        csv.writer(file).writerow([
            config.w_bits,
            config.a_bits,
            acc,
            metrics["compression_ratio"],
            metrics["compressed_mb"],
            metrics["weight_cr"],
            metrics["act_cr"],
            act_measured["act_cr_measured"],
        ])

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "quantized_acc", "goal": "maximize"},
    "parameters": {
        "w_bits": {"values": [8, 6, 4, 3, 2]},
        "a_bits": {"values": [8, 6, 4, 3, 2]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="CS6886-Quantization-Sweep")
    wandb.agent(sweep_id, run_sweep)