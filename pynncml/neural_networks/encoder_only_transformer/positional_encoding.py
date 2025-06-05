import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, window_size: int):
        super().__init__()

        pe = torch.zeros(window_size, d_model)
        position = torch.arange(0, window_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, window_size, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, d_model] where T = window_size
        pe = self.pe.to(x.device)  # [1, T, d_model]
        x = x + pe  # add position information
        return x
