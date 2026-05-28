"""Shared transformer encoder for SecUR."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        inv_freq = self.inv_freq[None, :, None].float()
        position_ids = position_ids[:, None, None].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + x_rot * sin


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool = True, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        kv = kv if kv is not None else x

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            cos = cos.unsqueeze(0).transpose(1, 2)  # (1, 1, T, head_dim)
            sin = sin.unsqueeze(0).transpose(1, 2)  # (1, 1, T, head_dim)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask[:, None, None, :]  # (B, 1, 1, T)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
        }[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(self.activation(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, causal: bool = True, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.attention = Attention(d_model, n_heads, causal=causal, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, activation=activation, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask=mask, cos=cos, sin=sin))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(config.d_model // config.n_heads, config.max_seq_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff,
                             causal=(config.encoder_attention == "causal"),
                             dropout=config.dropout, activation=config.activation)
            for _ in range(config.encoder_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_embedding(input_ids)

        seq_len = x.shape[1]
        cos, sin = self.rotary(x)
        cos = cos[:, :seq_len, :]
        sin = sin[:, :seq_len, :]

        for block in self.blocks:
            x = block(x, mask=attention_mask, cos=cos, sin=sin)

        return self.norm(x)
