"""Retrieval and generation metrics for SecBench."""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


# ── Retrieval Metrics ──

class RetrievalMetrics:
    @staticmethod
    def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        if not relevant:
            return 0.0
        retrieved_k = set(retrieved[:k])
        hits = len(relevant & retrieved_k)
        return hits / len(relevant)

    @staticmethod
    def precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        if k == 0:
            return 0.0
        retrieved_k = set(retrieved[:k])
        hits = len(relevant & retrieved_k)
        return hits / k

    @staticmethod
    def average_precision(relevant: Set[str], retrieved: List[str]) -> float:
        if not relevant:
            return 0.0
        score = 0.0
        hits = 0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                hits += 1
                score += hits / (i + 1)
        return score / len(relevant)

    @staticmethod
    def mean_reciprocal_rank(relevant: Set[str], retrieved: List[str]) -> float:
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(i + 2)

        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / ideal if ideal > 0 else 0.0

    @staticmethod
    def compute_all(relevant: Set[str], retrieved: List[str], ks: List[int] = None) -> Dict[str, float]:
        if ks is None:
            ks = [1, 3, 5, 10, 20]
        metrics = {}
        for k in ks:
            metrics[f"Recall@{k}"] = RetrievalMetrics.recall_at_k(relevant, retrieved, k)
            metrics[f"P@{k}"] = RetrievalMetrics.precision_at_k(relevant, retrieved, k)
            metrics[f"nDCG@{k}"] = RetrievalMetrics.ndcg_at_k(relevant, retrieved, k)
        metrics["MAP"] = RetrievalMetrics.average_precision(relevant, retrieved)
        metrics["MRR"] = RetrievalMetrics.mean_reciprocal_rank(relevant, retrieved)
        return metrics


# ── Generation Metrics ──

class GenerationMetrics:
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> float:
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    @staticmethod
    def token_f1(prediction: str, ground_truth: str) -> float:
        pred_tokens = set(prediction.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        if not gt_tokens:
            return 1.0 if not pred_tokens else 0.0
        if not pred_tokens:
            return 0.0
        intersection = pred_tokens & gt_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gt_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def rouge_l(prediction: str, ground_truth: str) -> Dict[str, float]:
        pred_words = prediction.lower().split()
        gt_words = ground_truth.lower().split()

        m = len(pred_words)
        n = len(gt_words)
        if m == 0 or n == 0:
            return {"rouge-l-f1": 0.0, "rouge-l-p": 0.0, "rouge-l-r": 0.0}

        # LCS using DP
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i - 1] == gt_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = dp[m][n]
        precision = lcs / m if m > 0 else 0.0
        recall = lcs / n if n > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"rouge-l-f1": f1, "rouge-l-p": precision, "rouge-l-r": recall}

    @staticmethod
    def ioc_extraction_precision_recall(
        predicted_iocs: Dict[str, List[str]],
        expected_iocs: Dict[str, List[str]],
    ) -> Dict[str, float]:
        all_pred = set()
        all_expected = set()
        exact_match = 0
        total_expected = 0

        for ioc_type, expected_list in expected_iocs.items():
            expected_set = set(e.lower() for e in expected_list)
            pred_set = set(p.lower() for p in predicted_iocs.get(ioc_type, []))
            all_pred.update(pred_set)
            all_expected.update(expected_set)
            exact_match += len(expected_set & pred_set)
            total_expected += len(expected_set)

        precision = exact_match / len(all_pred) if all_pred else 0.0
        recall = exact_match / total_expected if total_expected > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "ioc_precision": precision,
            "ioc_recall": recall,
            "ioc_f1": f1,
        }

    @staticmethod
    def cve_id_accuracy(prediction: str, expected_cves: List[str]) -> float:
        found = re.findall(r'CVE-\d{4}-\d{4,7}', prediction, re.IGNORECASE)
        expected_set = set(c.upper() for c in expected_cves)
        found_set = set(c.upper() for c in found)
        if not expected_set:
            return 1.0
        hits = len(expected_set & found_set)
        return hits / len(expected_set)

    @staticmethod
    def mitre_id_accuracy(prediction: str, expected_mitre: List[str]) -> float:
        found = re.findall(r'T\d{4}(?:\.\d{3})?', prediction)
        expected_set = set(expected_mitre)
        found_set = set(found)
        if not expected_set:
            return 1.0
        hits = len(expected_set & found_set)
        return hits / len(expected_set)

    @staticmethod
    def compute_all(
        prediction: str,
        ground_truth: str,
        expected_iocs: Optional[Dict[str, List[str]]] = None,
        expected_cves: Optional[List[str]] = None,
        expected_mitre: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        metrics = {
            "exact_match": GenerationMetrics.exact_match(prediction, ground_truth),
            "token_f1": GenerationMetrics.token_f1(prediction, ground_truth),
        }

        rouge = GenerationMetrics.rouge_l(prediction, ground_truth)
        metrics.update(rouge)

        if expected_iocs:
            pred_iocs = _extract_iocs_from_text(prediction)
            ioc_metrics = GenerationMetrics.ioc_extraction_precision_recall(pred_iocs, expected_iocs)
            metrics.update(ioc_metrics)

        if expected_cves:
            metrics["cve_accuracy"] = GenerationMetrics.cve_id_accuracy(prediction, expected_cves)

        if expected_mitre:
            metrics["mitre_accuracy"] = GenerationMetrics.mitre_id_accuracy(prediction, expected_mitre)

        return metrics


def _extract_iocs_from_text(text: str) -> Dict[str, List[str]]:
    return {
        "ip": re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text),
        "md5": re.findall(r'\b[a-fA-F0-9]{32}\b', text),
        "sha1": re.findall(r'\b[a-fA-F0-9]{40}\b', text),
        "sha256": re.findall(r'\b[a-fA-F0-9]{64}\b', text),
        "domain": re.findall(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b', text),
        "cve": re.findall(r'CVE-\d{4}-\d{4,7}', text, re.IGNORECASE),
        "url": re.findall(r'https?://[^\s]+', text),
        "port": re.findall(r'\bport\s+(\d{1,5})\b|:(\d{2,5})(?!\d)', text),
        "mitre": re.findall(r'T\d{4}(?:\.\d{3})?', text),
        "cwe": re.findall(r'CWE-\d{1,4}', text),
    }
