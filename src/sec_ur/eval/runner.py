"""Evaluation runner — connects SecBench to a trained SecUR model."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..config import SecURConfig
from ..model.sec_ur import SecURModel
from .secbench import SecBench
from .test_sets import get_all_sets


class SecBenchRunner:
    """Runs SecBench evaluation against a trained SecUR model."""

    def __init__(
        self,
        model: SecURModel,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        doc_store: Optional[Dict[str, str]] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.doc_store = doc_store or {}

        self.secbench = SecBench(
            retrieve_fn=self._retrieve,
            generate_fn=self._generate,
        )

    def set_document_store(self, store: Dict[str, str]) -> None:
        self.doc_store = store

    def _retrieve(self, query: str, top_k: int = 20) -> List[str]:
        if not self.doc_store:
            return []

        query_tokens = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        query_ids = query_tokens["input_ids"].to(self.device)
        query_mask = query_tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            query_vec = self.model.encode_queries(query_ids, query_mask)

        doc_ids = list(self.doc_store.keys())
        doc_texts = list(self.doc_store.values())

        scores = []
        for doc_text in doc_texts:
            doc_tokens = self.tokenizer(
                doc_text[:512],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            doc_ids_t = doc_tokens["input_ids"].to(self.device)
            doc_mask_t = doc_tokens["attention_mask"].to(self.device)

            with torch.no_grad():
                doc_vec = self.model.encode_documents(doc_ids_t, doc_mask_t)

            score = (query_vec @ doc_vec.T).item()
            scores.append(score)

        scored = list(zip(doc_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:top_k]]

    def _generate(self, query: str, context: str) -> str:
        input_text = context[:800] + "\n\n" + query if context else query

        tokens = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                attention_mask,
                max_new_tokens=256,
                temperature=0.7,
            )

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def run_all(self) -> Dict:
        return self.secbench.run(
            eval_retrieval=True,
            eval_generation=True,
            eval_end_to_end=True,
            top_k=20,
        )

    def run_category(self, category: str) -> Dict:
        from .test_sets import get_test_set

        self.secbench.test_sets = [get_test_set(category)]
        results = self.secbench.run(
            eval_retrieval=True,
            eval_generation=True,
            top_k=20,
        )
        self.secbench.test_sets = get_all_sets()
        return results

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer,
        doc_store: Optional[Dict[str, str]] = None,
    ) -> "SecBenchRunner":
        state = torch.load(checkpoint_path, map_location="cpu")
        if "config" in state:
            config = state["config"]
        else:
            config = SecURConfig()

        model = SecURModel(config)
        model.load_state_dict(state["model_state_dict"])

        return cls(model, tokenizer, doc_store=doc_store)
