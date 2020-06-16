import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Sequence
import torch.nn.functional as F

__all__ = ["MultiHeadAttention"]


class MultiHeadAttention(nn.Linear):
    def __init__(self, in_features: int, num_heads: int, bias: bool = True):
        super().__init__(in_features=in_features, out_features=in_features * num_heads, bias=bias)
        self.num_heads = num_heads

    @property
    def mean_attention_weights(self) -> Tensor:
        weight_unrolled = self.weight.data.view(self.num_heads, self.in_features, self.in_features)
        activated_diagonals = weight_unrolled.diagonal(dim1=1, dim2=2).softmax(1)
        output_mean = activated_diagonals.mean(0)

        return output_mean

    def forward(self, x: Tensor, freeze_inds: Optional[Sequence[int]] = None, return_softmax=False):
        out = super().forward(x)
        out = out.view(-1, self.num_heads, self.in_features).softmax(dim=-1)

        if freeze_inds is not None:
            freeze_inds_t = torch.as_tensor(freeze_inds, dtype=torch.long)
            out = out.clone()
            out[:, :, freeze_inds] = 1.0

        if not return_softmax:
            out = out * x.unsqueeze(1)

        return out.max(dim=1, keepdim=False)[0]
