"""Orchestrated data collection for SecUR training."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


def collect_all(
    data_dir: str = "data",
    nvd_api_key: Optional[str] = None,
    max_cves: int = 5000,
    include_mitre: bool = True,
    include_owasp: bool = True,
) -> Dict[str, int]:
    """Run all data collectors and return counts."""
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    # ── CVE ──
    print("[1/3] Collecting CVE data...")
    from sec_ur.data.cve_collector import CVECollector

    collector = CVECollector(data_dir=str(raw_dir / "cve"), api_key=nvd_api_key)
    cves = collector.fetch_recent(days_back=365 * 3, max_results=max_cves)
    parsed = []
    for entry in cves:
        p = collector.parse_cve_entry(entry)
        if p and p.get("description"):
            parsed.append(p)
    out = collector.save(parsed, "cve_data.json")
    counts["cve"] = len(parsed)
    print(f"  → {len(parsed)} CVEs → {out}")

    # ── MITRE ATT&CK ──
    if include_mitre:
        print("[2/3] Collecting MITRE ATT&CK data...")
        from sec_ur.data.mitre_collector import MITRECollector

        mitre = MITRECollector(data_dir=str(raw_dir / "mitre"))
        bundles = mitre.fetch_all()
        total = sum(len(v) for v in bundles.values())
        counts["mitre"] = total
        print(f"  → {total} objects across {len(bundles)} domains")

    # ── OWASP ──
    if include_owasp:
        print("[3/3] Collecting OWASP data...")
        from sec_ur.data.owasp_collector import OWASPCollector

        owasp = OWASPCollector(data_dir=str(raw_dir / "owasp"))
        top10 = owasp.fetch_top10()
        sheets = owasp.fetch_cheatsheets()
        counts["owasp"] = len(top10) + len(sheets)
        print(f"  → {len(top10)} top-10 items, {len(sheets)} cheatsheets")

    # Summary
    print(f"\n{'='*40}")
    print(f"Collection complete:")
    for k, v in counts.items():
        print(f"  {k}: {v} items")
    total = sum(counts.values())
    print(f"  Total: {total} items")
    print(f"{'='*40}")

    return counts


def build_training_metadata(data_dir: str = "data") -> str:
    """Build a metadata file describing the collected dataset."""
    raw_dir = Path(data_dir) / "raw"
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {},
    }

    total_docs = 0
    for json_file in sorted(raw_dir.rglob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            count = len(data) if isinstance(data, list) else 1
            total_docs += count
            meta["sources"][str(json_file.relative_to(raw_dir))] = count
        except Exception as e:
            meta["sources"][str(json_file.relative_to(raw_dir))] = f"error: {e}"

    meta["total_documents"] = total_docs

    out_path = Path(data_dir) / "dataset_metadata.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Metadata saved: {out_path}")
    print(f"Total documents: {total_docs}")

    return str(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect cybersecurity training data")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--nvd-key", default=None)
    parser.add_argument("--max-cves", type=int, default=5000)
    parser.add_argument("--skip-mitre", action="store_true")
    parser.add_argument("--skip-owasp", action="store_true")
    args = parser.parse_args()

    counts = collect_all(
        data_dir=args.data_dir,
        nvd_api_key=args.nvd_key,
        max_cves=args.max_cves,
        include_mitre=not args.skip_mitre,
        include_owasp=not args.skip_owasp,
    )

    build_training_metadata(args.data_dir)
