"""SecUR: Security Unified Retriever-Generator — end-to-end model."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import SecURConfig
from .encoder import Encoder
from .decoder import Decoder
from .heads import RetrievalHead


class SecURModel(nn.Module):
    """
    Unified encoder-decoder model for cybersecurity RAG.

    The shared encoder produces token representations used for:
      1. Dense retrieval (via the retrieval head)
      2. Cross-attention in the decoder for generation

    A single forward pass computes both retrieval and generation losses.
    """

    def __init__(self, config: SecURConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.retrieval_head = RetrievalHead(
            d_model=config.d_model,
            embedding_dim=config.embedding_dim,
            pooling=config.pooling,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_labels: Optional[torch.Tensor] = None,
        positive_ids: Optional[torch.Tensor] = None,
        negative_ids: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) query tokens
            attention_mask: (B, T) query attention mask
            decoder_input_ids: (B, T_dec) input for decoder (teacher-forcing)
            decoder_labels: (B, T_dec) target for generation loss
            positive_ids: (B, N_pos, T_pos) positive document tokens per query
            negative_ids: (B, N_neg, T_neg) negative document tokens per query
        """
        outputs = {}

        # Encode query
        query_encoding = self.encoder(input_ids, attention_mask)
        query_vec = self.retrieval_head(query_encoding, attention_mask)

        outputs["query_vec"] = query_vec

        # Retrieval loss — contrastive learning
        contrastive_loss = None
        if positive_ids is not None:
            contrastive_loss = self._compute_contrastive_loss(
                query_vec, query_encoding, attention_mask,
                positive_ids, positive_mask,
                negative_ids, negative_mask,
            )
            outputs["contrastive_loss"] = contrastive_loss

        # Generation loss
        generation_loss = None
        if decoder_input_ids is not None and decoder_labels is not None:
            gen_out = self.decoder(
                input_ids=decoder_input_ids,
                encoder_output=query_encoding,
                attention_mask=None,
                encoder_mask=attention_mask,
                labels=decoder_labels,
            )
            generation_loss = gen_out["loss"]
            outputs["generation_loss"] = generation_loss
            outputs["gen_logits"] = gen_out["logits"]

        # Combined loss
        if contrastive_loss is not None and generation_loss is not None:
            outputs["loss"] = (
                self.config.loss_alpha * contrastive_loss
                + self.config.loss_beta * generation_loss
            )
        elif contrastive_loss is not None:
            outputs["loss"] = contrastive_loss
        elif generation_loss is not None:
            outputs["loss"] = generation_loss

        return outputs

    def _compute_contrastive_loss(
        self,
        query_vec: torch.Tensor,
        query_encoding: torch.Tensor,
        query_mask: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_mask: Optional[torch.Tensor],
        negative_ids: Optional[torch.Tensor],
        negative_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if positive_ids.dim() == 2:
            positive_ids = positive_ids.unsqueeze(1)
            if positive_mask is not None:
                positive_mask = positive_mask.unsqueeze(1)

        B, N_pos, T_pos = positive_ids.shape
        pos_ids_flat = positive_ids.view(-1, T_pos)
        pos_mask_flat = positive_mask.view(-1, T_pos) if positive_mask is not None else None

        pos_encoding = self.encoder(pos_ids_flat, pos_mask_flat)
        pos_vecs = self.retrieval_head(pos_encoding, pos_mask_flat).view(B, N_pos, -1)

        if negative_ids is not None:
            if negative_ids.dim() == 2:
                negative_ids = negative_ids.unsqueeze(1)
                if negative_mask is not None:
                    negative_mask = negative_mask.unsqueeze(1)

            N_neg = negative_ids.shape[1]
            neg_ids_flat = negative_ids.view(-1, T_pos)
            neg_mask_flat = negative_mask.view(-1, T_pos) if negative_mask is not None else None
            neg_encoding = self.encoder(neg_ids_flat, neg_mask_flat)
            neg_vecs = self.retrieval_head(neg_encoding, neg_mask_flat).view(B, N_neg, -1)

            neg_vecs = neg_vecs.mean(dim=1)
        else:
            neg_vecs = None

        return self._ntxent_loss(query_vec, pos_vecs.mean(dim=1), neg_vecs)

    def _ntxent_loss(
        self,
        queries: torch.Tensor,
        positives: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalized temperature-scaled cross entropy loss."""
        temperature = self.config.contrastive_temperature

        sim = queries @ positives.T / temperature  # (B, B)
        labels = torch.arange(sim.shape[0], device=sim.device)

        if negatives is not None:
            neg_sim = queries @ negatives.T / temperature  # (B, B)
            sim = torch.cat([sim, neg_sim], dim=-1)

        loss = nn.functional.cross_entropy(sim, labels)
        return loss

    def encode_queries(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode queries into dense vectors for retrieval."""
        encoding = self.encoder(input_ids, attention_mask)
        return self.retrieval_head(encoding, attention_mask)

    def encode_documents(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode documents into dense vectors."""
        return self.encode_queries(input_ids, attention_mask)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> torch.Tensor:
        """Generate answer conditioned on input."""
        encoder_output = self.encoder(input_ids, attention_mask)
        return self.decoder.generate(
            encoder_output,
            encoder_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    @torch.no_grad()
    def retrieve_and_generate(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        doc_ids: torch.Tensor,
        doc_mask: torch.Tensor,
        top_k: int = 5,
        max_new_tokens: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end retrieval + generation.

        1. Encode query and documents
        2. Retrieve top-k documents by cosine similarity
        3. Concatenate and generate answer

        Returns: (answer_ids, doc_scores)
        """
        query_vec = self.encode_queries(query_ids, query_mask)
        doc_vecs = self.encode_documents(doc_ids, doc_mask)

        # Score: (B, N_docs)
        scores = query_vec @ doc_vecs.T

        # Retrieve top-k doc indices
        top_k = min(top_k, doc_ids.shape[0])
        top_scores, top_indices = scores.topk(top_k, dim=-1)

        # Concatenate retrieved doc text with query
        retrieved = doc_ids[top_indices[0]]  # (top_k, T_doc)
        flattened = retrieved.view(1, -1)

        # Generate with encoder
        combined = torch.cat([query_ids, flattened], dim=-1)[:, :self.config.max_seq_len]
        combined_mask = torch.ones_like(combined)

        return self.generate(combined, combined_mask, max_new_tokens=max_new_tokens), top_scores
