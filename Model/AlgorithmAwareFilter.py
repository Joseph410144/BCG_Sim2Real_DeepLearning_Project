"""Residual 1-D filter designed to preserve BCG input at initialization."""

import torch
from torch import nn


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(4 if channels % 4 == 0 else 1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4 if channels % 4 == 0 else 1, channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs):
        return self.activation(inputs + self.net(inputs))


class AlgorithmAwareResidualFilter(nn.Module):
    """Predict a bounded residual while exposing the residual for regularization."""

    def __init__(self, channels=32, blocks=6, dropout=0.05, max_residual_scale=1.0,
                 initial_gate=-2.0):
        super().__init__()
        self.input_projection = nn.Conv1d(1, channels, 7, padding=3)
        dilations = [2 ** (index % 4) for index in range(blocks)]
        self.blocks = nn.Sequential(*[
            DilatedResidualBlock(channels, dilation, dropout) for dilation in dilations
        ])
        self.output_projection = nn.Conv1d(channels, 1, 7, padding=3)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        # A modest initial gate preserves the input while avoiding the nearly
        # vanishing gradient produced by an extremely closed residual path.
        self.residual_gate = nn.Parameter(torch.tensor(float(initial_gate)))
        self.max_residual_scale = max_residual_scale

    def forward(self, inputs, return_residual=False):
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("Expected input shape [batch, 1, time]")
        features = self.blocks(self.input_projection(inputs))
        raw_residual = torch.tanh(self.output_projection(features))
        scale = torch.sigmoid(self.residual_gate) * self.max_residual_scale
        residual = scale * raw_residual
        output = inputs + residual
        return (output, residual) if return_residual else output
