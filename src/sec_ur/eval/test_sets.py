"""Pre-defined test sets for SecBench evaluation.

Each test set contains queries with known ground-truth documents
covering different cybersecurity capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TestExample:
    query: str
    relevant_doc_ids: List[str]
    ground_truth_answer: Optional[str] = None
    category: str = "general"
    metadata: Dict = field(default_factory=dict)


@dataclass
class TestSet:
    name: str
    description: str
    examples: List[TestExample]


# ── CVE Lookup ──

CVE_EXAMPLES = [
    TestExample(
        query="What is CVE-2024-3094 and why is it critical?",
        relevant_doc_ids=["CVE-2024-3094"],
        ground_truth_answer="A backdoor in XZ Utils liblzma (5.6.0/5.6.1) allowing remote code execution via SSH. CVSS 10.0.",
        category="cve_lookup",
    ),
    TestExample(
        query="Describe the Log4Shell vulnerability and its impact",
        relevant_doc_ids=["CVE-2021-44228"],
        ground_truth_answer="Remote code execution in Apache Log4j 2.x via JNDI lookups in log messages. CVSS 10.0, widespread exploitation.",
        category="cve_lookup",
    ),
    TestExample(
        query="What CVEs affect Apache HTTP Server 2.4.49?",
        relevant_doc_ids=["CVE-2021-41773", "CVE-2021-42013"],
        ground_truth_answer="Path traversal and RCE in Apache HTTP Server 2.4.49 (CVE-2021-41773) and 2.4.50 (CVE-2021-42013).",
        category="cve_lookup",
    ),
    TestExample(
        query="What is the CVSS score and attack vector for CVE-2023-34362?",
        relevant_doc_ids=["CVE-2023-34362"],
        ground_truth_answer="SQL injection in MOVEit Transfer. CVSS 9.8, network-based, low complexity, no privileges required.",
        category="cve_lookup",
    ),
    TestExample(
        query="Which Cisco vulnerabilities were disclosed in 2024?",
        relevant_doc_ids=["CVE-2024-20353", "CVE-2024-20354", "CVE-2024-20355"],
        category="cve_lookup",
    ),
]

# ── MITRE ATT&CK ──

MITRE_EXAMPLES = [
    TestExample(
        query="What MITRE technique covers file deletion as an anti-forensic tactic?",
        relevant_doc_ids=["T1070.004"],
        ground_truth_answer="T1070.004 — Indicator Removal: File Deletion. Part of the Defense Evasion tactic.",
        category="mitre_technique",
    ),
    TestExample(
        query="Describe the Privilege Escalation tactic techniques in MITRE ATT&CK",
        relevant_doc_ids=["TA0004"],
        ground_truth_answer="TA0004 covers techniques like Access Token Manipulation (T1134), Boot or Logon Autostart Execution (T1547), and Process Injection (T1055).",
        category="mitre_technique",
    ),
    TestExample(
        query="What is the MITRE ID for PowerShell-based execution?",
        relevant_doc_ids=["T1059.001"],
        ground_truth_answer="T1059.001 — Command and Scripting Interpreter: PowerShell.",
        category="mitre_technique",
    ),
    TestExample(
        query="How does credential dumping work according to MITRE?",
        relevant_doc_ids=["T1003"],
        ground_truth_answer="T1003 — OS Credential Dumping. Techniques to obtain credential material from OS and software.",
        category="mitre_technique",
    ),
    TestExample(
        query="What MITRE techniques are used for lateral movement via remote services?",
        relevant_doc_ids=["T1021", "T1021.001", "T1021.002", "T1021.003"],
        category="mitre_technique",
    ),
]

# ── Vulnerability Mitigation ──

MITIGATION_EXAMPLES = [
    TestExample(
        query="How to mitigate SQL injection in web applications?",
        relevant_doc_ids=["CWE-89", "OWASP-Top10-A03_2021"],
        ground_truth_answer="Use parameterized queries/prepared statements, input validation, least privilege DB accounts, and ORM frameworks.",
        category="mitigation",
    ),
    TestExample(
        query="What are the mitigations for cross-site scripting (XSS)?",
        relevant_doc_ids=["CWE-79", "OWASP-Top10-A03_2021"],
        ground_truth_answer="Output encoding, Content Security Policy (CSP), input validation, HttpOnly cookies, and XSS filters.",
        category="mitigation",
    ),
    TestExample(
        query="How to protect against Server-Side Request Forgery (SSRF)?",
        relevant_doc_ids=["CWE-918"],
        ground_truth_answer="Whitelist allowed URLs, disable unnecessary URL schemas, validate and sanitize user input, use network segmentation.",
        category="mitigation",
    ),
    TestExample(
        query="What mitigations exist for buffer overflow vulnerabilities?",
        relevant_doc_ids=["CWE-119", "CWE-120"],
        ground_truth_answer="Use memory-safe languages, stack canaries, ASLR, NX/DEP, bounds checking, and safe function libraries.",
        category="mitigation",
    ),
    TestExample(
        query="How to remediate insecure deserialization in Java?",
        relevant_doc_ids=["CWE-502"],
        ground_truth_answer="Use safe serialization formats (JSON), validate input, implement whitelist deserialization, use Java Serialization Filtering.",
        category="mitigation",
    ),
]

# ── IoC Extraction ──

IOC_EXAMPLES = [
    TestExample(
        query="Extract all IoCs from: Suspicious connection from 192.168.1.105 to 45.33.32.156:8443. File hash d41d8cd98f00b204e9800998ecf8427e detected on host CORP-WIN-01. Domain evil-malware.com resolved. CVE-2024-3094 exploitation attempt logged.",
        relevant_doc_ids=["ioc_patterns"],
        ground_truth_answer="IPs: 192.168.1.105, 45.33.32.156. Port: 8443. Hash: d41d8cd98f00b204e9800998ecf8427e. Domain: evil-malware.com. CVE: CVE-2024-3094.",
        category="ioc_extraction",
        metadata={"expected_iocs": {"ip": ["192.168.1.105", "45.33.32.156"], "hash": ["d41d8cd98f00b204e9800998ecf8427e"], "domain": ["evil-malware.com"], "cve": ["CVE-2024-3094"]}},
    ),
    TestExample(
        query="Find indicators in: Beacon callback to C2 server at https://malicious.xyz/c2 with SHA256 hash a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3. Src port 4444, dst port 80.",
        relevant_doc_ids=["ioc_patterns"],
        ground_truth_answer="URL: https://malicious.xyz/c2. Domain: malicious.xyz. SHA256: a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3. Ports: 4444, 80.",
        category="ioc_extraction",
        metadata={"expected_iocs": {"url": ["https://malicious.xyz/c2"], "domain": ["malicious.xyz"], "sha256": ["a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"]}},
    ),
    TestExample(
        query="What IoCs are present? Reg key HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Malware added. Process spawned cmd.exe /c net user hacker P@ssw0rd /add. T1059.001 technique observed.",
        relevant_doc_ids=["ioc_patterns"],
        ground_truth_answer="Registry key: HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Malware. Command: cmd.exe /c net user hacker P@ssw0rd /add. MITRE: T1059.001.",
        category="ioc_extraction",
        metadata={"expected_iocs": {"registry": ["HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Malware"], "command": ["cmd.exe /c net user hacker P@ssw0rd /add"], "mitre": ["T1059.001"]}},
    ),
    TestExample(
        query="Parse IoCs from: Malware sample MD5: 5d41402abc4b2a76b9719d911017c592. Communication with 10.0.0.5:53 (DNS tunneling). CWE-200 information disclosure.",
        relevant_doc_ids=["ioc_patterns"],
        ground_truth_answer="MD5: 5d41402abc4b2a76b9719d911017c592. IP: 10.0.0.5. Port: 53. CWE: CWE-200.",
        category="ioc_extraction",
        metadata={"expected_iocs": {"md5": ["5d41402abc4b2a76b9719d911017c592"], "ip": ["10.0.0.5"], "port": ["53"]}},
    ),
]

# ── Threat Intelligence ──

THREAT_INTEL_EXAMPLES = [
    TestExample(
        query="What is the difference between a worm and a trojan?",
        relevant_doc_ids=["malware_taxonomy"],
        ground_truth_answer="A worm self-replicates and spreads autonomously. A trojan disguises as legitimate software and requires user action.",
        category="threat_intel",
    ),
    TestExample(
        query="Describe the MITRE ATT&CK kill chain phases",
        relevant_doc_ids=["mitre_kill_chain"],
        ground_truth_answer="Reconnaissance, Resource Development, Initial Access, Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Collection, C2, Exfiltration, Impact.",
        category="threat_intel",
    ),
    TestExample(
        query="What is the Diamond Model of intrusion analysis?",
        relevant_doc_ids=["diamond_model"],
        ground_truth_answer="Four vertices: Adversary, Infrastructure, Capability, Victim. Relationships describe intrusion events.",
        category="threat_intel",
    ),
    TestExample(
        query="Explain the principle of least privilege",
        relevant_doc_ids=["security_principles"],
        ground_truth_answer="Users and processes should only have the minimum permissions needed to perform their function.",
        category="threat_intel",
    ),
    TestExample(
        query="What is defense in depth?",
        relevant_doc_ids=["security_principles"],
        ground_truth_answer="Multiple layers of security controls (network, host, application, data) so that if one fails, others still protect.",
        category="threat_intel",
    ),
]

# ── Compliance ──

COMPLIANCE_EXAMPLES = [
    TestExample(
        query="What are the PCI-DSS requirements for encryption of cardholder data?",
        relevant_doc_ids=["pci-dss-requirement-4"],
        ground_truth_answer="Requirement 4: Encrypt transmission of cardholder data across open/public networks. Use strong cryptography (TLS 1.2+).",
        category="compliance",
    ),
    TestExample(
        query="What ISO 27001 controls apply to access control?",
        relevant_doc_ids=["iso27001-A9"],
        ground_truth_answer="Annex A.9 covers access control: A.9.1 business requirements, A.9.2 user access management, A.9.3 user responsibilities, A.9.4 system and application access control.",
        category="compliance",
    ),
    TestExample(
        query="How does SOC 2 differ from SOC 1?",
        relevant_doc_ids=["soc2-overview"],
        ground_truth_answer="SOC 2 evaluates controls related to security, availability, processing integrity, confidentiality, and privacy. SOC 1 focuses on financial reporting controls.",
        category="compliance",
    ),
    TestExample(
        query="What are the five Trust Service Criteria of SOC 2?",
        relevant_doc_ids=["soc2-criteria"],
        ground_truth_answer="Security, Availability, Processing Integrity, Confidentiality, Privacy.",
        category="compliance",
    ),
]

# ── Security Code Review ──

CODE_REVIEW_EXAMPLES = [
    TestExample(
        query="Find vulnerabilities in this code: eval(request.GET.get('input'))",
        relevant_doc_ids=["CWE-95"],
        ground_truth_answer="CWE-95: Eval injection. Arbitrary code execution via unsanitized eval() of user input.",
        category="code_review",
    ),
    TestExample(
        query="Is this SQL query vulnerable?  cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')",
        relevant_doc_ids=["CWE-89"],
        ground_truth_answer="Yes, SQL injection (CWE-89). Use parameterized queries instead of f-string interpolation.",
        category="code_review",
    ),
    TestExample(
        query="What's wrong with: <img src='{{ user_input }}'> in Django template without escaping?",
        relevant_doc_ids=["CWE-79"],
        ground_truth_answer="Cross-site scripting (CWE-79). Django auto-escapes HTML, but image src with javascript: URLs bypasses this.",
        category="code_review",
    ),
]

# ── All test sets ──

ALL_TEST_SETS = [
    TestSet(
        name="cve_lookup",
        description="CVE identification and description retrieval",
        examples=CVE_EXAMPLES,
    ),
    TestSet(
        name="mitre_technique",
        description="MITRE ATT&CK technique identification",
        examples=MITRE_EXAMPLES,
    ),
    TestSet(
        name="mitigation",
        description="Cybersecurity mitigation and remediation advice",
        examples=MITIGATION_EXAMPLES,
    ),
    TestSet(
        name="ioc_extraction",
        description="Indicator of Compromise extraction from text",
        examples=IOC_EXAMPLES,
    ),
    TestSet(
        name="threat_intel",
        description="General threat intelligence knowledge",
        examples=THREAT_INTEL_EXAMPLES,
    ),
    TestSet(
        name="compliance",
        description="Compliance and regulatory framework knowledge",
        examples=COMPLIANCE_EXAMPLES,
    ),
    TestSet(
        name="code_review",
        description="Security code review and vulnerability identification",
        examples=CODE_REVIEW_EXAMPLES,
    ),
]


def get_test_set(name: str) -> TestSet:
    for ts in ALL_TEST_SETS:
        if ts.name == name:
            return ts
    raise ValueError(f"Unknown test set: {name}. Available: {[t.name for t in ALL_TEST_SETS]}")


def get_all_sets() -> List[TestSet]:
    return ALL_TEST_SETS
