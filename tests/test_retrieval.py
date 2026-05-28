"""Tests for the dataset and training pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader

from sec_ur.config import SecURConfig
from sec_ur.data.dataset import CyberSecurityDataset, collate_cyber_batch


@pytest.fixture
def sample_data(tmp_path):
    import json
    data = [
        {
            "id": "CVE-2024-12345",
            "description": "A critical buffer overflow in Apache HTTP Server 2.4.49",
            "weaknesses": ["CWE-119"],
            "references": [{"url": "https://example.com"}],
            "base_score": 9.8,
            "severity": "CRITICAL",
        },
        {
            "id": "CVE-2024-67890",
            "description": "SQL injection in WordPress plugin",
            "weaknesses": ["CWE-89"],
            "references": [],
            "base_score": 7.5,
            "severity": "HIGH",
        },
    ]
    f = tmp_path / "test_cves.json"
    with open(f, "w") as fh:
        json.dump(data, fh)
    return str(f)


class MockTokenizer:
    def __call__(self, text, **kwargs):
        length = kwargs.get("max_length", 128)
        ret = kwargs.get("return_tensors")
        out = {
            "input_ids": torch.randint(0, 100, (1, length)),
            "attention_mask": torch.ones(1, length),
        }
        if ret == "pt" or True:
            return {k: v for k, v in out.items()}
        return out


def test_dataset_initialization(sample_data):
    ds = CyberSecurityDataset(
        data_path=sample_data,
        tokenizer_fn=MockTokenizer(),
        max_query_len=32,
        max_doc_len=64,
        num_negatives=2,
    )
    assert len(ds) > 0


def test_dataset_item_shape(sample_data):
    ds = CyberSecurityDataset(
        data_path=sample_data,
        tokenizer_fn=MockTokenizer(),
        max_query_len=32,
        max_doc_len=64,
        num_negatives=2,
    )
    item = ds[0]
    assert "input_ids" in item
    assert "positive_ids" in item
    assert "negative_ids" in item
    assert "decoder_input_ids" in item
    assert "decoder_labels" in item


def test_collate_function(sample_data):
    ds = CyberSecurityDataset(
        data_path=sample_data,
        tokenizer_fn=MockTokenizer(),
        max_query_len=32,
        max_doc_len=64,
        num_negatives=1,
    )
    batch = [ds[0], ds[1]]
    collated = collate_cyber_batch(batch)
    assert collated["input_ids"].shape[0] == 2  # batch size
    assert collated["negative_ids"].shape[0] == 2
    assert collated["negative_ids"].shape[1] == 1  # num negatives


def test_dataloader(sample_data):
    ds = CyberSecurityDataset(
        data_path=sample_data,
        tokenizer_fn=MockTokenizer(),
        max_query_len=32,
        max_doc_len=64,
        num_negatives=1,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_cyber_batch)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] <= 2
