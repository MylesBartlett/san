import torch.nn as nn
from torch import Tensor

__all__ = ["MultiHeadAttention"]


class MultiHeadAttention(nn.Linear):
    def __init__(self, in_features: int, num_heads: int, bias: bool = True):
        super().__init__(in_features=in_features, out_features=in_features * num_heads, bias=bias)
        self.num_heads = num_heads

    def forward(self, x: Tensor, return_softmax=False):
        out = super().forward(x)
        out = out.view(-1, self.num_heads, self.in_features).softmax(dim=-1)
        if not return_softmax:
            out = out * x.unsqueeze(1)

        return out.max(dim=1, keepdim=False)[0]

    @property
    def mean_attention_weights(self) -> Tensor:
        weight_unrolled = self.weight.data.view(self.num_heads, self.in_channels, self.in_channels)
        activated_diagonals = weight_unrolled.diagonal(dim1=1, dim2=2).softmax(1)
        output_mean = activated_diagonals.mean(0)

        return output_mean
