"""OWASP and general security documentation collector."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from bs4 import BeautifulSoup


OWASP_BASES = [
    "https://owasp.org/www-project-top-ten/",
    "https://cheatsheetseries.owasp.org/",
    "https://owasp.org/www-project-application-security-verification-standard/",
]

CWE_API = "https://cwe-api.mitre.org/api/v1/cwe/"


class OWASPCollector:
    def __init__(self, data_dir: str = "data/raw/owasp"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=60.0, follow_redirects=True)

    def fetch_top10(self) -> List[Dict]:
        response = self.client.get("https://owasp.org/www-project-top-ten/")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        items = []
        for header in soup.find_all(["h2", "h3"]):
            text = header.get_text(strip=True)
            if re.match(r"^A\d{2}", text):
                content = []
                for sibling in header.find_next_siblings():
                    if sibling.name and sibling.name.startswith("h"):
                        break
                    content.append(sibling.get_text(strip=True))
                items.append({
                    "title": text,
                    "content": " ".join(content),
                    "source": "OWASP Top 10",
                })

        path = self.data_dir / "owasp_top10.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        return items

    def fetch_cheatsheets(self) -> List[Dict]:
        index_url = "https://cheatsheetseries.owasp.org/"
        response = self.client.get(index_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        sheets = []
        for link in soup.select("a[href*='cheatsheets']"):
            url = link.get("href", "")
            if url.endswith(".html") and "index" not in url:
                name = link.get_text(strip=True)
                if name:
                    try:
                        full_url = url if url.startswith("http") else f"{index_url}{url.lstrip('/')}"
                        sheet_response = self.client.get(full_url)
                        sheet_soup = BeautifulSoup(sheet_response.text, "lxml")
                        content = sheet_soup.get_text(separator="\n", strip=True)
                        sheets.append({
                            "title": name,
                            "content": content[:5000],
                            "url": full_url,
                            "source": "OWASP Cheatsheet",
                        })
                    except Exception:
                        continue

        path = self.data_dir / "owasp_cheatsheets.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sheets, f, indent=2)
        return sheets


class CWECollector:
    def __init__(self, data_dir: str = "data/raw/cwe"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=30.0, follow_redirects=True)

    def fetch_cwe(self, cwe_id: str) -> Optional[Dict]:
        response = self.client.get(f"{CWE_API}{cwe_id}")
        if response.status_code != 200:
            return None
        data = response.json()

        return {
            "id": f"CWE-{cwe_id}",
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "extended_description": data.get("extended_description", ""),
            "likelihood": data.get("likelihood_of_exploit", ""),
            "detection_methods": [
                m.get("method", "") for m in data.get("detection_methods", [])
            ],
            "potential_mitigations": [
                m.get("description", "") for m in data.get("potential_mitigations", [])
            ],
        }
