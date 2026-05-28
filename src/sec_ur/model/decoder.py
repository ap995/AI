"""Decoder with cross-attention to encoder outputs for generation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Attention, FeedForward, RotaryEmbedding, TransformerBlock


class CrossAttentionLayer(nn.Module):
    """Decoder layer: self-attn → cross-attn (to encoder) → FF."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.self_attn = Attention(d_model, n_heads, causal=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = Attention(d_model, n_heads, causal=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, activation=activation, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=self_mask, cos=cos, sin=sin))
        x = x + self.dropout(self.cross_attn(self.norm2(x), kv=encoder_output, mask=cross_mask))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(config.d_model // config.n_heads, config.max_seq_len)

        self.layers = nn.ModuleList([
            CrossAttentionLayer(config.d_model, config.n_heads, config.d_ff,
                                dropout=config.dropout, activation=config.activation)
            for _ in range(config.decoder_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings and LM head
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        x = self.token_embedding(input_ids)

        seq_len = x.shape[1]
        cos, sin = self.rotary(x)
        cos = cos[:, :seq_len, :]
        sin = sin[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x, encoder_output, self_mask=attention_mask, cross_mask=encoder_mask,
                      cos=cos, sin=sin)

        x = self.norm(x)

        if labels is not None:
            logits = self.lm_head(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return {"loss": loss, "logits": logits}

        return {"logits": self.lm_head(x)}

    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self(generated, encoder_output, encoder_mask=encoder_mask)["logits"]
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k, dim=-1)
                next_logits[next_logits < values[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if (next_token == eos_token_id).all():
                break

        return generated
