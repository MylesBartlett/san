from typing import Sequence, Tuple, Union

import numpy as np
import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention

__all__ = ["SANNetwork"]


class SANNetwork(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes: Union[int, Sequence[int]],
        hidden_layer_size,
        dropout=0.02,
        num_heads=2,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.multi_head = MultiHeadAttention(in_channels, num_heads=num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_layer_size),
            nn.Dropout(dropout),
            nn.SELU(),
            nn.Linear(hidden_layer_size, np.sum(num_classes)),
        )

        self.in_channels = in_channels
        self.num_targets = 1 if isinstance(num_classes, int) else len(num_classes)
        self.num_classes = num_classes
        self.num_heads = num_heads

    def split_outputs(self, outputs: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        if self.num_targets > 1:
            outputs = outputs.split(self.num_classes, dim=1)
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        out = self.multi_head(x)
        out = self.classifier(out)

        return out
