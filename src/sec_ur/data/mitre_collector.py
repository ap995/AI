"""MITRE ATT&CK data collector."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import httpx


ATTACK_ENTERPRISE_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
ATTACK_MOBILE_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/mobile-attack/mobile-attack.json"
ATTACK_ICS_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/ics-attack/ics-attack.json"


class MITRECollector:
    def __init__(self, data_dir: str = "data/raw/mitre"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=120.0, follow_redirects=True)

    def fetch_all(self) -> Dict[str, List[Dict]]:
        return {
            "enterprise": self._fetch_bundle(ATTACK_ENTERPRISE_URL, "enterprise"),
            "mobile": self._fetch_bundle(ATTACK_MOBILE_URL, "mobile"),
            "ics": self._fetch_bundle(ATTACK_ICS_URL, "ics"),
        }

    def _fetch_bundle(self, url: str, domain: str) -> List[Dict]:
        response = self.client.get(url)
        response.raise_for_status()
        bundle = response.json()

        objects = bundle.get("objects", [])
        parsed = []
        for obj in objects:
            p = self._parse_stix_object(obj, domain)
            if p:
                parsed.append(p)

        path = self.data_dir / f"{domain}_attack.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)

        return parsed

    def _parse_stix_object(self, obj: Dict, domain: str) -> Optional[Dict]:
        obj_type = obj.get("type", "")
        if obj_type not in ("attack-pattern", "course-of-action", "malware", "tool", "intrusion-set", "campaign", "x-mitre-tactic", "x-mitre-matrix"):
            return None

        base = {
            "id": obj.get("id", ""),
            "attack_id": "",
            "type": obj_type,
            "name": obj.get("name", ""),
            "description": obj.get("description", ""),
            "domain": domain,
        }

        external_refs = obj.get("external_references", [])
        for ref in external_refs:
            if ref.get("source_name") in ("mitre-attack", "mitre-mobile-attack", "mitre-ics-attack"):
                base["attack_id"] = ref.get("external_id", "")
                break

        if obj_type == "attack-pattern":
            kill_chain = obj.get("kill_chain_phases", [])
            base["tactic"] = [k["phase_name"] for k in kill_chain if "phase_name" in k]

        if obj_type == "x-mitre-tactic":
            base["attack_id"] = obj.get("x_mitre_shortname", "").upper()
            base["name"] = obj.get("name", "")

        if obj_type == "malware":
            base["aliases"] = obj.get("aliases", [])

        base["labels"] = obj.get("labels", [])

        return base
