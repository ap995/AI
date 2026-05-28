"""Cybersecurity dataset — builds contrastive pairs and generation targets."""

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class CyberSecurityDataset(Dataset):
    """
    Dataset that produces (query, positive_doc, negative_docs, generation_target) tuples
    from cybersecurity sources.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_fn: Callable,
        max_query_len: int = 128,
        max_doc_len: int = 512,
        num_negatives: int = 7,
        split: str = "train",
    ):
        self.data = self._load_data(data_path)
        self.tokenizer_fn = tokenizer_fn
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.num_negatives = num_negatives
        self.split = split

        self.pairs = self._build_pairs()

    def _load_data(self, path: str) -> List[Dict]:
        p = Path(path)
        all_data = []
        if p.is_dir():
            for f in sorted(p.glob("*.json")):
                with open(f, "r", encoding="utf-8") as fh:
                    try:
                        data = json.load(fh)
                        if isinstance(data, list):
                            all_data.extend(data)
                        elif isinstance(data, dict):
                            all_data.append(data)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON: {f}")
        elif p.is_file():
            with open(p, "r", encoding="utf-8") as fh:
                all_data = json.load(fh)
        return all_data

    def _build_pairs(self) -> List[Dict]:
        """Build (query, positive, negative_indices) from raw data."""
        pairs = []

        for item in self.data:
            source_type = item.get("type", item.get("source", "general"))

            if "id" in item and "description" in item:
                cve_id = item["id"]
                desc = item["description"]
                weaknesses = item.get("weaknesses", [])
                refs = item.get("references", [])
                ref_urls = [r.get("url", "") for r in refs if r.get("url")]

                # Query: natural question about the CVE
                query = f"Tell me about {cve_id}: {desc[:200]}"
                positive = desc

                # Add mitigation queries
                if weaknesses:
                    for weakness in weaknesses[:3]:
                        pairs.append({
                            "query": f"How to mitigate {weakness}?",
                            "positive": desc,
                            "negative_keys": ["description"],
                            "source": source_type,
                        })

                pairs.append({
                    "query": query,
                    "positive": positive,
                    "negative_keys": ["description"],
                    "source": source_type,
                })

            elif "attack_id" in item and "description" in item:
                attack_id = item["attack_id"]
                name = item.get("name", "")
                desc = item.get("description", "")
                tactic = item.get("tactic", [])

                query = f"What is {attack_id} ({name})?"
                if tactic:
                    query = f"Describe the {', '.join(tactic)} technique {attack_id}: {name}"

                pairs.append({
                    "query": query,
                    "positive": f"{name}: {desc[:500]}",
                    "negative_keys": ["description", "name"],
                    "source": source_type,
                })

            elif "title" in item and "content" in item:
                title = item["title"]
                content = item.get("content", "")
                pairs.append({
                    "query": title,
                    "positive": content[:500],
                    "negative_keys": ["content", "title"],
                    "source": source_type,
                })

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        query_text = pair["query"]
        positive_text = pair["positive"]

        # Hard negatives: sample from other items
        query_tokens = self.tokenizer_fn(
            query_text, padding="max_length", truncation=True, max_length=self.max_query_len,
            return_tensors="pt",
        )

        pos_tokens = self.tokenizer_fn(
            positive_text, padding="max_length", truncation=True, max_length=self.max_doc_len,
            return_tensors="pt",
        )

        neg_indices = random.sample(
            [i for i in range(len(self.pairs)) if i != idx],
            min(self.num_negatives, len(self.pairs) - 1),
        )
        neg_texts = [self.pairs[i]["positive"][:self.max_doc_len] for i in neg_indices]

        neg_tokens_list = []
        for nt in neg_texts:
            tokens = self.tokenizer_fn(
                nt, padding="max_length", truncation=True, max_length=self.max_doc_len,
                return_tensors="pt",
            )
            neg_tokens_list.append(tokens["input_ids"].squeeze(0))

        neg_tokens = torch.stack(neg_tokens_list) if neg_tokens_list else torch.zeros((1, self.max_doc_len), dtype=torch.long)

        gen_text = pair.get("generation_target", positive_text[:256])
        gen_tokens = self.tokenizer_fn(
            gen_text, padding="max_length", truncation=True, max_length=256,
            return_tensors="pt",
        )

        return {
            "input_ids": query_tokens["input_ids"].squeeze(0),
            "attention_mask": query_tokens["attention_mask"].squeeze(0),
            "positive_ids": pos_tokens["input_ids"].squeeze(0),
            "positive_mask": pos_tokens["attention_mask"].squeeze(0),
            "negative_ids": neg_tokens,
            "negative_mask": torch.ones_like(neg_tokens),
            "decoder_labels": gen_tokens["input_ids"].squeeze(0),
            "decoder_input_ids": torch.cat([
                torch.full((1,), 1, dtype=torch.long),
                gen_tokens["input_ids"].squeeze(0)[:-1],
            ]),
        }


def collate_cyber_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for CyberSecurityDataset batches."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key == "negative_ids":
            max_neg = max(b["negative_ids"].shape[0] for b in batch)
            max_len = max(b["negative_ids"].shape[1] for b in batch)
            padded = []
            for b in batch:
                neg = b[key]
                n, l = neg.shape
                pad = torch.zeros((max_neg, max_len), dtype=neg.dtype)
                pad[:n, :l] = neg
                padded.append(pad)
            result[key] = torch.stack(padded)
        elif key == "negative_mask":
            max_neg = max(b["negative_mask"].shape[0] for b in batch)
            max_len = max(b["negative_mask"].shape[1] for b in batch)
            padded = []
            for b in batch:
                m = b[key]
                n, l = m.shape
                pad = torch.zeros((max_neg, max_len), dtype=m.dtype)
                pad[:n, :l] = m
                padded.append(pad)
            result[key] = torch.stack(padded)
        else:
            result[key] = torch.stack([b[key] for b in batch])
    return result
