import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [B, h, T, d_k]
        mask:    [B, 1, T, T] or None
        Returns: [B, h, T, d_k]
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, h, T, T]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_scores = F.softmax(scores, dim=-1)  # [B, h, T, T]
        attention_scores = self.dropout(attention_scores)

        return torch.matmul(attention_scores, V), attention_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [B, T, d_model]
        Returns: [B, T, d_model]
        """
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)  # [B, h, T, d_k]
        K = self.W_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        out, attention_scores = self.attention(Q, K, V, mask)         # [B, h, T, d_k]

        out = out.transpose(1, 2).contiguous().view(B, T, -1)          # [B, T, d_model]
        out = self.W_o(out)                                            # [B, T, d_model]
        return self.dropout(out)
