"""SecBench: Main evaluation harness for cybersecurity RAG models."""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .test_sets import TestExample, TestSet, get_all_sets
from .metrics import RetrievalMetrics, GenerationMetrics


class SecBench:
    """
    Cybersecurity benchmark for RAG systems.

    Evaluates both retrieval quality and generation accuracy
    across 7 cybersecurity test categories.
    """

    def __init__(
        self,
        retrieve_fn: Optional[Callable] = None,
        generate_fn: Optional[Callable] = None,
        test_sets: Optional[List[TestSet]] = None,
    ):
        """
        Args:
            retrieve_fn: callable(query: str, top_k: int) -> List[str] of doc IDs
            generate_fn: callable(query: str, context: str) -> str
            test_sets: list of TestSet to evaluate (default: all)
        """
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn
        self.test_sets = test_sets or get_all_sets()
        self.results: Dict[str, Dict] = {}

    def set_retriever(self, fn: Callable) -> None:
        self.retrieve_fn = fn

    def set_generator(self, fn: Callable) -> None:
        self.generate_fn = fn

    def evaluate_retrieval(self, test_set: TestSet, top_k: int = 20) -> Dict[str, float]:
        all_metrics = {}
        n = 0

        for example in test_set.examples:
            if not self.retrieve_fn:
                continue
            retrieved = self.retrieve_fn(example.query, top_k=top_k)
            relevant = set(example.relevant_doc_ids)

            metrics = RetrievalMetrics.compute_all(relevant, retrieved)
            for k, v in metrics.items():
                all_metrics.setdefault(k, 0.0)
                all_metrics[k] += v
            n += 1

        if n > 0:
            return {k: v / n for k, v in all_metrics.items()}
        return {}

    def evaluate_generation(self, test_set: TestSet) -> Dict[str, float]:
        all_metrics = {}
        n = 0

        for example in test_set.examples:
            if not self.generate_fn or not example.ground_truth_answer:
                continue

            prediction = self.generate_fn(example.query, "")
            gt = example.ground_truth_answer
            metadata = example.metadata or {}

            metrics = GenerationMetrics.compute_all(
                prediction=prediction,
                ground_truth=gt,
                expected_iocs=metadata.get("expected_iocs"),
                expected_cves=metadata.get("expected_cves"),
                expected_mitre=metadata.get("expected_mitre"),
            )
            for k, v in metrics.items():
                all_metrics.setdefault(k, 0.0)
                all_metrics[k] += v
            n += 1

        if n > 0:
            return {k: v / n for k, v in all_metrics.items()}
        return {}

    def evaluate_retrieve_and_generate(
        self,
        test_set: TestSet,
        top_k: int = 5,
    ) -> Dict[str, float]:
        all_metrics = {}
        n = 0

        for example in test_set.examples:
            if not self.retrieve_fn or not self.generate_fn:
                continue

            retrieved = self.retrieve_fn(example.query, top_k=top_k)
            relevant = set(example.relevant_doc_ids)

            ret_metrics = RetrievalMetrics.compute_all(relevant, retrieved)
            all_metrics["retrieval_recall_at_5"] = all_metrics.get("retrieval_recall_at_5", 0.0) + ret_metrics.get("Recall@5", 0.0)
            all_metrics["retrieval_ndcg_at_5"] = all_metrics.get("retrieval_ndcg_at_5", 0.0) + ret_metrics.get("nDCG@5", 0.0)

            if example.ground_truth_answer:
                context = "\n".join(retrieved[:3]) if retrieved else ""
                prediction = self.generate_fn(example.query, context)
                gt = example.ground_truth_answer

                gen_metrics = GenerationMetrics.compute_all(
                    prediction=prediction,
                    ground_truth=gt,
                )
                for k, v in gen_metrics.items():
                    all_metrics.setdefault(k, 0.0)
                    all_metrics[k] += v
            n += 1

        if n > 0:
            return {k: v / n for k, v in all_metrics.items()}
        return {}

    def run(
        self,
        eval_retrieval: bool = True,
        eval_generation: bool = True,
        eval_end_to_end: bool = False,
        top_k: int = 20,
    ) -> Dict[str, Dict[str, float]]:
        results = {}

        for test_set in self.test_sets:
            print(f"\n{'='*60}")
            print(f"Test Set: {test_set.name}")
            print(f"  {test_set.description}")
            print(f"  Examples: {len(test_set.examples)}")
            print(f"{'='*60}")

            set_results = {}

            if eval_retrieval and self.retrieve_fn:
                ret_metrics = self.evaluate_retrieval(test_set, top_k=top_k)
                if ret_metrics:
                    set_results["retrieval"] = ret_metrics
                    print(f"  [Retrieval] R@5: {ret_metrics.get('Recall@5', 0):.4f}  "
                          f"nDCG@5: {ret_metrics.get('nDCG@5', 0):.4f}  "
                          f"MRR: {ret_metrics.get('MRR', 0):.4f}")

            if eval_generation and self.generate_fn:
                gen_metrics = self.evaluate_generation(test_set)
                if gen_metrics:
                    set_results["generation"] = gen_metrics
                    print(f"  [Generation] ROUGE-L: {gen_metrics.get('rouge-l-f1', 0):.4f}  "
                          f"Token F1: {gen_metrics.get('token_f1', 0):.4f}  "
                          f"EM: {gen_metrics.get('exact_match', 0):.4f}")

            if eval_end_to_end:
                e2e_metrics = self.evaluate_retrieve_and_generate(test_set, top_k=5)
                if e2e_metrics:
                    set_results["end_to_end"] = e2e_metrics
                    print(f"  [E2E] ROUGE-L: {e2e_metrics.get('rouge-l-f1', 0):.4f}")

            results[test_set.name] = set_results

        self.results = results

        print(f"\n{'='*60}")
        print(f"SecBench Complete")
        print(f"Categories evaluated: {len(results)}")
        print(f"{'='*60}")

        return results

    def summary(self) -> Dict[str, float]:
        """Aggregate results across all test sets."""
        agg = {}
        count = {}

        for set_name, set_results in self.results.items():
            for eval_type, metrics in set_results.items():
                for metric_name, value in metrics.items():
                    key = f"{eval_type}/{metric_name}"
                    agg[key] = agg.get(key, 0.0) + value
                    count[key] = count.get(key, 0) + 1

        return {k: v / count[k] for k, v in agg.items()}

    def save_report(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "results": self.results,
            "summary": self.summary(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(p, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Report saved: {p}")
        return str(p)

    def print_summary_table(self) -> None:
        summary = self.summary()
        print(f"\n{'='*60}")
        print(f"SecBench Summary")
        print(f"{'='*60}")

        categories = sorted(set(k.split("/")[0] for k in summary))
        for cat in categories:
            print(f"\n[{cat}]")
            for k, v in sorted(summary.items()):
                if k.startswith(f"{cat}/"):
                    metric = k.split("/", 1)[1]
                    print(f"  {metric:20s}: {v:.4f}")
