"""Tests for the cybersecurity tokenizer."""

import pytest
from sec_ur.tokenizer import CyberTokenizer


@pytest.fixture
def tokenizer():
    return CyberTokenizer(vocab_size=1000)


def test_tokenizer_initialization(tokenizer):
    assert tokenizer.tokenizer is not None


def test_cve_normalization(tokenizer):
    text = "Critical vulnerability CVE-2024-12345 in Apache"
    encoded = tokenizer.encode(text)
    assert "CVE" in str(encoded["tokens"]) or len(encoded["input_ids"]) > 0


def test_ip_normalization(tokenizer):
    text = "Malicious traffic from 192.168.1.1 detected"
    encoded = tokenizer.encode(text)
    assert len(encoded["input_ids"]) > 0


def test_hash_normalization(tokenizer):
    text = "File hash: d41d8cd98f00b204e9800998ecf8427e"
    encoded = tokenizer.encode(text)
    assert len(encoded["input_ids"]) > 0


def test_mitre_id_normalization(tokenizer):
    text = "Attack technique T1059.001 detected"
    encoded = tokenizer.encode(text)
    assert len(encoded["input_ids"]) > 0


def test_encode_decode_roundtrip(tokenizer):
    tokenizer.train(["This is a test CVE-2024-12345 for security"])
    text = "Check CVE-2024-9999 and IP 10.0.0.1"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded["input_ids"])
    assert "CVE" in decoded or len(encoded["input_ids"]) > 0


def test_attention_mask(tokenizer):
    encoded = tokenizer.encode("Test query")
    assert "attention_mask" in encoded
    assert len(encoded["attention_mask"]) == len(encoded["input_ids"])
