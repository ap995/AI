"""Document processing and chunking utilities for cybersecurity data."""

import re
from typing import Dict, List, Optional


def clean_cyber_text(text: str) -> str:
    """Clean security text while preserving CVE IDs, IPs, hashes, etc."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def semantic_chunk(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    """Chunk security documents at semantic boundaries (paragraphs, sections)."""
    paragraphs = text.split('\n\n')
    chunks = []
    current = []

    for para in paragraphs:
        current.append(para)
        current_text = '\n\n'.join(current)

        if len(current_text) >= max_chars:
            chunks.append(current_text)
            # Keep last paragraph for overlap
            current = [current[-1]] if len(current[-1]) < max_chars else []

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def extract_iocs(text: str) -> Dict[str, List[str]]:
    """Extract Indicators of Compromise from text."""
    iocs = {
        "ipv4": re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text),
        "md5": re.findall(r'\b[a-fA-F0-9]{32}\b', text),
        "sha1": re.findall(r'\b[a-fA-F0-9]{40}\b', text),
        "sha256": re.findall(r'\b[a-fA-F0-9]{64}\b', text),
        "domain": re.findall(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b', text),
        "cve": re.findall(r'CVE-\d{4}-\d{4,7}', text, re.IGNORECASE),
        "url": re.findall(r'https?://[^\s]+', text),
    }
    return {k: list(set(v)) for k, v in iocs.items()}


def extract_mitre_techniques(text: str) -> List[str]:
    """Extract MITRE ATT&CK technique IDs from text."""
    return re.findall(r'T\d{4}(?:\.\d{3})?', text)


def format_cve_entry(entry: Dict) -> Dict:
    """Format a CVE entry into a training-ready document."""
    return {
        "id": entry.get("id", ""),
        "title": entry.get("id", ""),
        "content": entry.get("description", ""),
        "metadata": {
            "severity": entry.get("severity"),
            "score": entry.get("base_score"),
            "weaknesses": entry.get("weaknesses", []),
        },
        "type": "cve",
    }
