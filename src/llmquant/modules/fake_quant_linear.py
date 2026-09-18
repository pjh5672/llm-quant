import torch.nn as nn
import torch.nn.functional as F

from llmquant.modifiers.quantization.scheme import QuantizationScheme
from llmquant.utils.quant_ops import fake_quantize


class FakeQuantLinear(nn.Module):
    """Weights stored already dequantized; activations fake-quantized per forward."""

    def __init__(self, weight, bias, scheme: QuantizationScheme):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = bias
        self.scheme = scheme

    @classmethod
    def from_linear(cls, linear: nn.Linear, scheme: QuantizationScheme):
        w = linear.weight.detach()
        w = fake_quantize(w, scheme.weights) if scheme.weights else w.clone()
        return cls(w, linear.bias, scheme)

    def forward(self, x):
        if self.scheme.input_activations:
            x = fake_quantize(x, self.scheme.input_activations)
        return F.linear(x, self.weight, self.bias)
