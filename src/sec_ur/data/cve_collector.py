"""CVE data collector from NVD API."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httpx


CVE_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"


class CVECollector:
    def __init__(self, data_dir: str = "data/raw/cve", api_key: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.client = httpx.Client(timeout=60.0, follow_redirects=True)

    def fetch_recent(self, days_back: int = 30, max_results: int = 5000) -> List[Dict]:
        start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        end_date = datetime.utcnow().isoformat()

        params = {
            "pubStartDate": start_date,
            "pubEndDate": end_date,
            "resultsPerPage": min(max_results, 2000),
            "startIndex": 0,
        }
        return self._fetch_all(params, max_results)

    def fetch_by_year(self, year: int) -> List[Dict]:
        params = {
            "pubStartDate": f"{year}-01-01T00:00:00.000",
            "pubEndDate": f"{year}-12-31T23:59:59.999",
            "resultsPerPage": 2000,
            "startIndex": 0,
        }
        return self._fetch_all(params, 20000)

    def _fetch_all(self, params: Dict, max_results: int) -> List[Dict]:
        all_vulns = []
        headers = {}
        if self.api_key:
            headers["apiKey"] = self.api_key

        while len(all_vulns) < max_results:
            response = self.client.get(CVE_API_BASE, params=params, headers=headers)
            if response.status_code == 403:
                raise PermissionError("NVD API access denied. Provide an API key or reduce request rate.")
            response.raise_for_status()

            data = response.json()
            vulnerabilities = data.get("vulnerabilities", [])
            all_vulns.extend(vulnerabilities)

            total_results = data.get("totalResults", 0)
            params["startIndex"] += len(vulnerabilities)

            if params["startIndex"] >= total_results or not vulnerabilities:
                break

            time.sleep(0.6)

        return all_vulns[:max_results]

    def save(self, vulns: List[Dict], filename: str) -> str:
        path = self.data_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vulns, f, indent=1, default=str)
        return str(path)

    def parse_cve_entry(self, entry: Dict) -> Optional[Dict]:
        try:
            cve = entry.get("cve", {})
            cve_id = cve.get("id", "")
            descriptions = cve.get("descriptions", [])
            description = next(
                (d["value"] for d in descriptions if d.get("lang") == "en"), ""
            )

            metrics = cve.get("metrics", {})
            cvss_v31 = metrics.get("cvssMetricV31", [])
            base_score = None
            severity = None
            if cvss_v31:
                base_score = cvss_v31[0]["cvssData"].get("baseScore")
                severity = cvss_v31[0]["cvssData"].get("baseSeverity")

            cvss_v30 = metrics.get("cvssMetricV30", [])
            if cvss_v30 and base_score is None:
                base_score = cvss_v30[0]["cvssData"].get("baseScore")
                severity = cvss_v30[0]["cvssData"].get("baseSeverity")

            references = [
                {"url": ref.get("url"), "source": ref.get("source")}
                for ref in cve.get("references", [])[:10]
            ]

            weaknesses = []
            for weakness in cve.get("weaknesses", []):
                for desc in weakness.get("description", []):
                    if desc.get("value", "").startswith("CWE-"):
                        weaknesses.append(desc["value"])

            return {
                "id": cve_id,
                "description": description,
                "base_score": base_score,
                "severity": severity,
                "weaknesses": weaknesses,
                "references": references,
                "published": cve.get("published"),
                "last_modified": cve.get("lastModified"),
            }
        except Exception:
            return None

    def fetch_full_history(self) -> List[Dict]:
        all_cves = []
        for year in range(1999, datetime.utcnow().year + 1):
            print(f"Fetching CVEs for {year}...")
            year_cves = self.fetch_by_year(year)
            parsed = []
            for entry in year_cves:
                p = self.parse_cve_entry(entry)
                if p and p["description"]:
                    parsed.append(p)
            all_cves.extend(parsed)
            print(f"  -> {len(parsed)} CVEs parsed")
        return all_cves
