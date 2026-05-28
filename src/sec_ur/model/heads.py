"""Pooling and projection heads for retrieval embeddings."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolingHead(nn.Module):
    """Converts encoder token-level outputs to a single vector for retrieval."""

    def __init__(self, d_model: int, embedding_dim: int, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Linear(d_model, embedding_dim)

    def forward(self, encoder_output: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = encoder_output.mean(dim=1)
        elif self.pooling == "cls":
            pooled = encoder_output[:, 0]
        elif self.pooling == "weighted":
            if attention_mask is not None:
                weights = torch.sigmoid(self.proj(encoder_output))
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (encoder_output * weights * mask).sum(dim=1) / (weights * mask).sum(dim=1).clamp(min=1)
            else:
                weights = torch.sigmoid(self.proj(encoder_output))
                pooled = (encoder_output * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return F.normalize(self.proj(pooled), p=2, dim=-1)


class RetrievalHead(nn.Module):
    """Combined pooling + optional linear projection."""

    def __init__(self, d_model: int, embedding_dim: int, pooling: str = "mean"):
        super().__init__()
        self.pool = PoolingHead(d_model, embedding_dim, pooling=pooling)

    def forward(self, encoder_output: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.pool(encoder_output, attention_mask)


class ClassificationHead(nn.Module):
    """Optional head for security classification tasks."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        pooled = encoder_output.mean(dim=1)
        return self.fc(pooled)
