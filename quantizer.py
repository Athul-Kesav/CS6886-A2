import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_ste(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return x
    if bits < 1:
        raise ValueError("bits must be at least 1")

    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max().clamp_min(1e-8) / max(-qmin, qmax)
    quantized = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    return x + (quantized - x).detach()


class QuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        weight_bits=8,
        act_bits=8,
        padding_mode="zeros",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.weight_bits = weight_bits
        self.act_bits = act_bits

    def forward(self, input):
        return self._conv(
            quantize_ste(input, self.act_bits),
            quantize_ste(self.weight, self.weight_bits),
        )

    def _conv(self, x, weight):
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, weight_bits=8, act_bits=8):
        super().__init__(in_features, out_features, bias)
        self.weight_bits = weight_bits
        self.act_bits = act_bits

    def forward(self, input):
        return F.linear(
            quantize_ste(input, self.act_bits),
            quantize_ste(self.weight, self.weight_bits),
            self.bias,
        )


def _make_quantized_conv(conv, weight_bits, act_bits):
    quantized = QuantizedConv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None,
        weight_bits,
        act_bits,
        conv.padding_mode,
    )
    with torch.no_grad():
        quantized.weight.copy_(conv.weight)
        if conv.bias is not None and quantized.bias is not None:
            quantized.bias.copy_(conv.bias)
    return quantized


def fuse_conv_bn(conv, bn):
    if bn.training:
        raise ValueError("BatchNorm2d must be in eval mode")

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        True,
        conv.padding_mode,
    )
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)

    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        if fused.bias is not None:
            fused.bias.copy_(scale * (bias - bn.running_mean) + bn.bias)
    return fused


def apply_quantization(module: nn.Module, weight_bits=8, act_bits=8):
    children = list(module.named_children())
    index = 0

    while index < len(children):
        name, child = children[index]

        if (
            isinstance(child, nn.Conv2d)
            and index + 1 < len(children)
            and isinstance(children[index + 1][1], nn.BatchNorm2d)
        ):
            next_name, bn = children[index + 1]
            setattr(module, name, _make_quantized_conv(fuse_conv_bn(child, bn), weight_bits, act_bits))
            setattr(module, next_name, nn.Identity())
            index += 2
            continue

        if isinstance(child, nn.Conv2d):
            setattr(module, name, _make_quantized_conv(child, weight_bits, act_bits))
        elif isinstance(child, nn.Linear):
            quantized = QuantizedLinear(
                child.in_features,
                child.out_features,
                child.bias is not None,
                weight_bits,
                act_bits,
            )
            with torch.no_grad():
                quantized.weight.copy_(child.weight)
                if child.bias is not None:
                    quantized.bias.copy_(child.bias)
            setattr(module, name, quantized)
        else:
            apply_quantization(child, weight_bits, act_bits)

        index += 1