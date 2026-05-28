"""Tests for SecBench evaluation framework."""

import pytest
from sec_ur.eval.test_sets import TestExample, TestSet, get_test_set, get_all_sets
from sec_ur.eval.metrics import RetrievalMetrics, GenerationMetrics


# ── Test Sets ──

def test_all_test_sets_have_examples():
    sets = get_all_sets()
    assert len(sets) >= 5
    for ts in sets:
        assert len(ts.examples) > 0, f"{ts.name} has no examples"
        for ex in ts.examples:
            assert ex.query, f"Example in {ts.name} has no query"
            assert ex.relevant_doc_ids, f"Example in {ts.name} has no relevant_doc_ids"


def test_get_test_set_by_name():
    ts = get_test_set("cve_lookup")
    assert ts.name == "cve_lookup"
    assert len(ts.examples) > 0


def test_get_test_set_invalid():
    with pytest.raises(ValueError):
        get_test_set("nonexistent")


def test_cve_examples_have_ids():
    ts = get_test_set("cve_lookup")
    for ex in ts.examples:
        for doc_id in ex.relevant_doc_ids:
            assert doc_id.startswith("CVE-") or "CVE" in doc_id


def test_mitre_examples_have_techniques():
    ts = get_test_set("mitre_technique")
    for ex in ts.examples:
        for doc_id in ex.relevant_doc_ids:
            assert doc_id.startswith("TA") or doc_id.startswith("T")


# ── Retrieval Metrics ──

def test_recall_at_k():
    relevant = {"a", "b"}
    retrieved = ["a", "c", "b", "d"]
    assert RetrievalMetrics.recall_at_k(relevant, retrieved, 1) == 0.5
    assert RetrievalMetrics.recall_at_k(relevant, retrieved, 2) == 0.5
    assert RetrievalMetrics.recall_at_k(relevant, retrieved, 3) == 1.0


def test_precision_at_k():
    relevant = {"a", "b"}
    retrieved = ["a", "c", "b", "d"]
    assert RetrievalMetrics.precision_at_k(relevant, retrieved, 1) == 1.0
    assert RetrievalMetrics.precision_at_k(relevant, retrieved, 2) == 0.5


def test_average_precision():
    relevant = {"a", "b"}
    retrieved = ["a", "c", "b", "d"]
    ap = RetrievalMetrics.average_precision(relevant, retrieved)
    # At pos 1: 1/1 = 1. At pos 3: 2/3. AP = (1 + 2/3) / 2 = 0.8333
    assert abs(ap - 0.8333) < 0.01


def test_mean_reciprocal_rank():
    relevant = {"b"}
    retrieved = ["a", "b", "c"]
    assert RetrievalMetrics.mean_reciprocal_rank(relevant, retrieved) == 0.5


def test_mrr_no_relevant():
    relevant = {"x"}
    retrieved = ["a", "b", "c"]
    assert RetrievalMetrics.mean_reciprocal_rank(relevant, retrieved) == 0.0


def test_ndcg_at_k():
    relevant = {"a", "b"}
    retrieved = ["a", "c", "b", "d"]
    ndcg = RetrievalMetrics.ndcg_at_k(relevant, retrieved, 5)
    assert ndcg > 0 and ndcg <= 1.0


def test_compute_all():
    relevant = {"a", "b"}
    retrieved = ["a", "c", "b", "d"]
    metrics = RetrievalMetrics.compute_all(relevant, retrieved, ks=[1, 3])
    assert "Recall@1" in metrics
    assert "nDCG@3" in metrics
    assert "MAP" in metrics
    assert "MRR" in metrics
    assert metrics["MAP"] > 0


# ── Generation Metrics ──

def test_exact_match():
    assert GenerationMetrics.exact_match("hello world", "hello world") == 1.0
    assert GenerationMetrics.exact_match("Hello World", "hello world") == 1.0
    assert GenerationMetrics.exact_match("hello", "world") == 0.0


def test_token_f1():
    f1 = GenerationMetrics.token_f1("cat dog mouse", "cat dog elephant")
    assert f1 > 0 and f1 < 1.0
    assert GenerationMetrics.token_f1("same text", "same text") == 1.0
    assert GenerationMetrics.token_f1("completely different", "no match here") == 0.0


def test_rouge_l_identical():
    rouge = GenerationMetrics.rouge_l("the cat sat on the mat", "the cat sat on the mat")
    assert rouge["rouge-l-f1"] == 1.0


def test_rouge_l_partial():
    rouge = GenerationMetrics.rouge_l("the cat sat", "the cat sat on the mat")
    assert rouge["rouge-l-p"] == 1.0
    assert rouge["rouge-l-r"] < 1.0
    assert rouge["rouge-l-f1"] > 0


def test_ioc_precision_recall():
    pred = {
        "ip": ["192.168.1.1", "10.0.0.1"],
        "hash": ["abc123"],
    }
    expected = {
        "ip": ["192.168.1.1", "10.0.0.2"],
        "hash": ["abc123", "def456"],
    }
    metrics = GenerationMetrics.ioc_extraction_precision_recall(pred, expected)
    assert metrics["ioc_precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["ioc_recall"] == pytest.approx(2.0 / 4.0)
    assert metrics["ioc_f1"] > 0


def test_cve_id_accuracy():
    pred = "CVE-2024-3094 and CVE-2021-44228 are critical"
    expected = ["CVE-2024-3094", "CVE-2021-44228"]
    assert GenerationMetrics.cve_id_accuracy(pred, expected) == 1.0


def test_cve_id_partial():
    pred = "CVE-2024-3094 is bad"
    expected = ["CVE-2024-3094", "CVE-2021-44228"]
    assert GenerationMetrics.cve_id_accuracy(pred, expected) == 0.5


def test_mitre_id_accuracy():
    pred = "T1059.001 and T1003 techniques detected"
    expected = ["T1059.001", "T1003"]
    assert GenerationMetrics.mitre_id_accuracy(pred, expected) == 1.0


def test_compute_all_basic():
    metrics = GenerationMetrics.compute_all(
        prediction="CVE-2024-3094 is critical",
        ground_truth="CVE-2024-3094 is a critical vulnerability",
    )
    assert "exact_match" in metrics
    assert "token_f1" in metrics
    assert "rouge-l-f1" in metrics


def test_compute_all_with_iocs():
    metrics = GenerationMetrics.compute_all(
        prediction="IP 192.168.1.1 and hash abc123 found",
        ground_truth="Found IP 192.168.1.1 and MD5 hash abc123",
        expected_iocs={"ip": ["192.168.1.1"], "md5": ["abc123"]},
    )
    assert "ioc_f1" in metrics


def test_compute_all_with_cves():
    metrics = GenerationMetrics.compute_all(
        prediction="CVE-2024-3094",
        ground_truth="CVE-2024-3094",
        expected_cves=["CVE-2024-3094"],
    )
    assert metrics["cve_accuracy"] == 1.0


def test_compute_all_with_mitre():
    metrics = GenerationMetrics.compute_all(
        prediction="Technique T1059.001",
        ground_truth="T1059.001",
        expected_mitre=["T1059.001"],
    )
    assert metrics["mitre_accuracy"] == 1.0


def test_ioc_extraction_empty():
    pred = {"ip": [], "md5": [], "domain": []}
    expected = {"ip": [], "md5": [], "domain": []}
    metrics = GenerationMetrics.ioc_extraction_precision_recall(pred, expected)
    assert isinstance(metrics["ioc_f1"], float)
    assert metrics["ioc_precision"] == 0.0


# ── End to End ──

def test_retrieval_metrics_consistency():
    """When all relevant docs are retrieved, recall should be 1.0."""
    relevant = {"doc1", "doc2", "doc3"}
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    metrics = RetrievalMetrics.compute_all(relevant, retrieved, ks=[1, 3, 5])
    assert metrics["Recall@5"] == 1.0
    assert metrics["Recall@3"] == 1.0
    assert metrics["nDCG@5"] == 1.0
