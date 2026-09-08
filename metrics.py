import torch
import torch.nn as nn

from quantizer import QuantizedConv2d, QuantizedLinear


def calculate_compression_metrics(model, weight_bits, act_bits):
    layers = tuple(
        layer
        for layer in model.modules()
        if isinstance(layer, (QuantizedConv2d, QuantizedLinear, nn.Conv2d, nn.Linear))
    )
    parameters = sum(
        layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
        for layer in layers
    )
    baseline_mb = parameters * 32 / (8 * 1024 * 1024)
    compressed_mb = (
        parameters * weight_bits + len(layers) * 32
    ) / (8 * 1024 * 1024)

    return {
        "baseline_mb": baseline_mb,
        "compressed_mb": compressed_mb,
        "compression_ratio": baseline_mb / compressed_mb if compressed_mb else 1.0,
        "weight_cr": 32.0 / weight_bits,
        "act_cr": 32.0 / act_bits,
    }


def measure_activation_footprint(model, dataloader, device, act_bits, num_batches=1):
    total_elements = 0

    def record_activation(_, __, output):
        nonlocal total_elements
        total_elements += output.numel()

    handles = [
        layer.register_forward_hook(record_activation)
        for layer in model.modules()
        if isinstance(layer, (QuantizedConv2d, QuantizedLinear))
    ]
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for batch_index, (inputs, _) in enumerate(dataloader):
                model(inputs.to(device))
                if batch_index + 1 >= num_batches:
                    break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    fp32_mb = total_elements * 32 / (8 * 1024 * 1024)
    quant_mb = total_elements * act_bits / (8 * 1024 * 1024)

    return {
        "activation_elements": total_elements,
        "activation_fp32_mb": fp32_mb,
        "activation_quant_mb": quant_mb,
        "act_cr_measured": fp32_mb / quant_mb if quant_mb else 1.0,
    }