"""Tests for the SecUR model architecture."""

import pytest
import torch

from sec_ur.config import SecURConfig
from sec_ur.model.sec_ur import SecURModel


@pytest.fixture
def config():
    return SecURConfig(
        vocab_size=1000,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        n_heads=2,
        d_ff=128,
        max_seq_len=128,
        embedding_dim=64,
    )


@pytest.fixture
def model(config):
    return SecURModel(config)


def test_model_initialization(model):
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.retrieval_head is not None


def test_forward_pass_query_only(model):
    batch = {
        "input_ids": torch.randint(0, 100, (2, 32)),
        "attention_mask": torch.ones(2, 32),
    }
    outputs = model(**batch)
    assert "query_vec" in outputs
    assert outputs["query_vec"].shape == (2, 64)


def test_forward_with_generation(model):
    batch = {
        "input_ids": torch.randint(0, 100, (2, 32)),
        "attention_mask": torch.ones(2, 32),
        "decoder_input_ids": torch.randint(0, 100, (2, 32)),
        "decoder_labels": torch.randint(0, 100, (2, 32)),
    }
    outputs = model(**batch)
    assert "generation_loss" in outputs
    assert "gen_logits" in outputs


def test_forward_with_contrastive(model):
    batch = {
        "input_ids": torch.randint(0, 100, (2, 32)),
        "attention_mask": torch.ones(2, 32),
        "positive_ids": torch.randint(0, 100, (2, 1, 64)),
        "positive_mask": torch.ones(2, 1, 64),
        "negative_ids": torch.randint(0, 100, (2, 3, 64)),
        "negative_mask": torch.ones(2, 3, 64),
    }
    outputs = model(**batch)
    assert "contrastive_loss" in outputs


def test_full_forward(model):
    batch = {
        "input_ids": torch.randint(0, 100, (2, 32)),
        "attention_mask": torch.ones(2, 32),
        "positive_ids": torch.randint(0, 100, (2, 1, 64)),
        "positive_mask": torch.ones(2, 1, 64),
        "negative_ids": torch.randint(0, 100, (2, 3, 64)),
        "negative_mask": torch.ones(2, 3, 64),
        "decoder_input_ids": torch.randint(0, 100, (2, 32)),
        "decoder_labels": torch.randint(0, 100, (2, 32)),
    }
    outputs = model(**batch)
    assert "loss" in outputs
    assert "contrastive_loss" in outputs
    assert "generation_loss" in outputs


def test_encode_queries(model, config):
    input_ids = torch.randint(0, 100, (2, 32))
    mask = torch.ones(2, 32)
    vecs = model.encode_queries(input_ids, mask)
    assert vecs.shape == (2, config.embedding_dim)
    # Unit normalized
    assert torch.allclose(vecs.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_generate(model):
    input_ids = torch.randint(0, 100, (1, 32))
    mask = torch.ones(1, 32)
    output = model.generate(input_ids, mask, max_new_tokens=10)
    assert output.shape[1] > 1  # at least BOS + 1 token


def test_model_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
    print(f"Model parameters: {total:,}")


def test_retrieve_and_generate(model, config):
    query = torch.randint(0, 100, (1, 32))
    q_mask = torch.ones(1, 32)
    docs = torch.randint(0, 100, (10, 64))
    d_mask = torch.ones(10, 64)
    answers, scores = model.retrieve_and_generate(query, q_mask, docs, d_mask, top_k=3)
    assert answers.shape[1] > 1
    assert scores.shape == (1, 3)
