import torch
import torch.nn as nn
from .attention_mechanism import MultiHeadAttention

class LayerNormalization(nn.Module):
    def __init__(self, d_model=512, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))   # scale
        self.bias = nn.Parameter(torch.zeros(d_model))   # shift
        self.eps = eps

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)           # [B, T, 1]
        std = x.std(dim=-1, keepdim=True)             # [B, T, 1]
        normalized = (x - mean) / (std + self.eps)    # [B, T, d_model]
        return self.alpha * normalized + self.bias    # [B, T, d_model]

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()  # or nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        x: [B, T, d_model]
        sublayer: a function/module that returns a tensor of the same shape
        """
        return self.norm(x + self.dropout(sublayer(x)))

# ------------------------
# Encoder Block
# ------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, h=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)

        self.residual1 = ResidualConnection(d_model=d_model, dropout=dropout)
        self.residual2 = ResidualConnection(d_model=d_model, dropout=dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.mha(x, mask))
        x = self.residual2(x, self.ffn)
        return x

# ------------------------
# Full Encoder (stacked blocks)
# ------------------------
class Encoder(nn.Module):
    def __init__(self, num_layers=4, d_model=512, h=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, h, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x