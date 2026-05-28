from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SecURConfig:
    # Model architecture
    vocab_size: int = 32_000
    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    encoder_attention: str = "causal"  # causal or bidirectional

    # Embedding head (for retrieval)
    embedding_dim: int = 512
    pooling: str = "mean"  # mean, cls, weighted

    # Training
    batch_size: int = 16
    grad_accum_steps: int = 4
    learning_rate: float = 5e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    contrastive_margin: float = 0.2
    contrastive_temperature: float = 0.05
    loss_alpha: float = 1.0  # retrieval loss weight
    loss_beta: float = 1.0  # generation loss weight

    # Data
    max_doc_len: int = 512
    max_query_len: int = 128
    num_negatives: int = 7

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "data/checkpoints"

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # Cybersecurity-specific
    cve_vocab_size: int = 2048  # extra tokens for CVE IDs
    ioc_vocab_size: int = 1024  # extra tokens for common IoC patterns
