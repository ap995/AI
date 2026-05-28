"""Contrastive and generation loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """NT-Xent loss with in-batch negatives."""

    def __init__(self, temperature: float = 0.05, margin: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, query_vec: torch.Tensor, doc_vec: torch.Tensor) -> torch.Tensor:
        sim = query_vec @ doc_vec.T / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device)
        return F.cross_entropy(sim, labels)


class TripletLoss(nn.Module):
    """Triplet margin loss for anchor-positive-negative tuples."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_dist = (anchor - positive).pow(2).sum(-1)
        neg_dist = (anchor - negative).pow(2).sum(-1)
        return F.relu(pos_dist - neg_dist + self.margin).mean()


class CombinedLoss(nn.Module):
    """Weighted combination of contrastive retrieval and generation loss."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, temperature: float = 0.05):
        super().__init__()
        self.contrastive = ContrastiveLoss(temperature=temperature)
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        query_vec: torch.Tensor,
        doc_vec: torch.Tensor,
        gen_logits: torch.Tensor,
        gen_labels: torch.Tensor,
    ) -> dict:
        ret_loss = self.contrastive(query_vec, doc_vec)
        gen_loss = F.cross_entropy(
            gen_logits.view(-1, gen_logits.size(-1)),
            gen_labels.view(-1),
            ignore_index=-100,
        )
        total = self.alpha * ret_loss + self.beta * gen_loss
        return {
            "loss": total,
            "retrieval_loss": ret_loss,
            "generation_loss": gen_loss,
        }
