"""Cybersecurity-aware tokenizer with CVE, IP, hash, and IoC pattern handling."""

import re
from typing import List, Optional, Tuple

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing


CVE_PATTERN = r"CVE-\d{4}-\d{4,7}"
IP_PATTERN = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
HASH_PATTERN = r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b"
DOMAIN_PATTERN = r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b"
PORT_PATTERN = r"\bport\s+\d{1,5}\b|:\d{2,5}\b(?!\d)"
MITRE_PATTERN = r"(?:T\d{4}(?:\.\d{3})?)|(?:TA\d{4})"
CWE_PATTERN = r"CWE-\d{1,4}"
CAPEC_PATTERN = r"CAPEC-\d{1,4}"

SEC_PATTERNS = {
    "CVE": re.compile(CVE_PATTERN, re.IGNORECASE),
    "IP": re.compile(IP_PATTERN),
    "MD5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "SHA1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
    "SHA256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    "DOMAIN": re.compile(DOMAIN_PATTERN, re.IGNORECASE),
    "PORT": re.compile(PORT_PATTERN, re.IGNORECASE),
    "MITRE_ID": re.compile(MITRE_PATTERN),
    "CWE": re.compile(CWE_PATTERN),
    "CAPEC": re.compile(CAPEC_PATTERN),
}


class CyberTokenizer:
    def __init__(self, vocab_size: int = 32_000):
        self.vocab_size = vocab_size
        self.tokenizer = self._build_base_tokenizer()

    def _build_base_tokenizer(self) -> Tokenizer:
        tok = Tokenizer(models.BPE())
        tok.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Replace(r"\r\n", "\n"),
        ])
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                r"(\s+|(?=[^\w\s])|(?<=[^\w\s]))",
                behavior="isolated",
            ),
        ])
        tok.decoder = decoders.BPEDecoder()

        tok.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[PAD]", 0),
                ("[BOS]", 1),
                ("[EOS]", 2),
                ("[UNK]", 3),
            ],
        )
        return tok

    def train(self, texts: List[str]) -> None:
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self.tokenizer.train_from_iterator(self._iter_pretokenize(texts), trainer)

    def _iter_pretokenize(self, texts: List[str]):
        for text in texts:
            yield self._normalize_security_tokens(text)

    def _normalize_security_tokens(self, text: str) -> str:
        for name, pattern in SEC_PATTERNS.items():
            placeholder = f" ☐{name}☐ "
            text = pattern.sub(placeholder, text)
        return text

    def _restore_security_tokens(self, text: str) -> str:
        for name in reversed(list(SEC_PATTERNS.keys())):
            text = text.replace(f"☐{name}☐", f"[{name}]")
        return text

    def encode(self, text: str, add_special_tokens: bool = True) -> dict:
        normalized = self._normalize_security_tokens(text)
        encoded = self.tokenizer.encode(normalized, add_special_tokens=add_special_tokens)
        return {
            "input_ids": encoded.ids,
            "attention_mask": encoded.attention_mask,
            "tokens": encoded.tokens,
        }

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return self._restore_security_tokens(text)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @vocab_size.setter
    def vocab_size(self, value: int) -> None:
        pass

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "CyberTokenizer":
        tok = cls.__new__(cls)
        tok.tokenizer = Tokenizer.from_file(path)
        return tok
