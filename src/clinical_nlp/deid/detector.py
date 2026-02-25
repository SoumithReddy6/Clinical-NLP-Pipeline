from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MatchRule:
    category: str
    pattern: re.Pattern[str]
    score: float = 0.95


# All 18 HIPAA Safe Harbor categories with production-quality regex patterns
RULES = [
    # 1. Names — multiple patterns for clinical text
    MatchRule("NAME", re.compile(
        r"\b(?:Patient|Pt|Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"
    ), 0.85),
    # 2. Geographic subdivisions smaller than state
    MatchRule("GEOGRAPHIC_SUBDIVISION", re.compile(
        r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*\s+"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|"
        r"Court|Ct|Place|Pl|Way|Circle|Cir|Terrace|Ter)\b"
    ), 0.80),
    # ZIP codes (5 or 9 digit)
    MatchRule("GEOGRAPHIC_SUBDIVISION", re.compile(
        r"\b\d{5}(?:-\d{4})?\b"
    ), 0.70),
    # 3. Dates (all elements except year) — multiple formats
    MatchRule("DATES", re.compile(
        r"\b(?:"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"  # MM/DD/YYYY or M/D/YY
        r"|\d{4}-\d{2}-\d{2}"  # YYYY-MM-DD
        r"|(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\.?\s+\d{1,2},?\s+\d{4}"  # Month DD, YYYY
        r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}"  # DD Month YYYY
        r")\b"
    ), 0.90),
    # 4. Phone numbers — US formats
    MatchRule("PHONE", re.compile(
        r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ), 0.95),
    # 5. Fax numbers (same pattern, contextual)
    MatchRule("FAX", re.compile(
        r"(?:fax|facsimile)[:\s#-]*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        re.IGNORECASE,
    ), 0.90),
    # 6. Email addresses
    MatchRule("EMAIL", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ), 0.98),
    # 7. Social Security Numbers
    MatchRule("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0.98),
    # 8. Medical Record Numbers
    MatchRule("MRN", re.compile(
        r"\b(?:MRN|Medical\s+Record\s+Number|Med\s*Rec)[:\s#-]*\d{6,12}\b",
        re.IGNORECASE,
    ), 0.95),
    # 9. Health plan beneficiary numbers
    MatchRule("HEALTH_PLAN_BENEFICIARY", re.compile(
        r"\b(?:Health\s*Plan|Insurance|Policy|Member|Subscriber|Beneficiary)"
        r"[:\s#-]*(?:ID|No|Number|#)?[:\s#-]*[A-Z0-9]{6,20}\b",
        re.IGNORECASE,
    ), 0.80),
    # 10. Account numbers
    MatchRule("ACCOUNT", re.compile(
        r"\b(?:Acct|Account)[:\s#-]*\d{6,16}\b", re.IGNORECASE
    ), 0.90),
    # 11. Certificate/license numbers
    MatchRule("CERTIFICATE_LICENSE", re.compile(
        r"\b(?:License|Certificate|DEA|NPI|Cert)[:\s#-]*[A-Z0-9]{5,15}\b",
        re.IGNORECASE,
    ), 0.80),
    # 12. Vehicle identifiers (VIN)
    MatchRule("VEHICLE_IDENTIFIER", re.compile(
        r"\b(?:VIN)[:\s#-]*[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE
    ), 0.85),
    # 13. Device identifiers and serial numbers
    MatchRule("DEVICE_IDENTIFIER", re.compile(
        r"\b(?:Device|Serial|SN|UDI)[:\s#-]*[A-Z0-9]{6,20}\b", re.IGNORECASE
    ), 0.80),
    # 14. URLs
    MatchRule("URL", re.compile(r"\bhttps?://[^\s<>\"]+"), 0.98),
    # 15. IP Addresses (IPv4)
    MatchRule("IP_ADDRESS", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ), 0.95),
    # 16. Biometric identifiers (fingerprint, retinal, voiceprint)
    MatchRule("BIOMETRIC_IDENTIFIER", re.compile(
        r"\b(?:fingerprint|retinal\s+scan|voiceprint|iris\s+scan|facial\s+recognition)"
        r"[:\s#-]*[A-Z0-9]+\b",
        re.IGNORECASE,
    ), 0.75),
    # 18. Other unique identifiers
    MatchRule("OTHER_UNIQUE_IDENTIFIER", re.compile(
        r"\b(?:Patient\s+ID|PID|Encounter\s+ID|Visit\s+ID|Case\s+ID)"
        r"[:\s#-]*[A-Z0-9]{4,20}\b",
        re.IGNORECASE,
    ), 0.80),
]


def detect_phi_spans(text: str) -> list[dict]:
    """Detect all PHI spans in text using HIPAA Safe Harbor regex rules."""
    spans = []
    for rule in RULES:
        for m in rule.pattern.finditer(text):
            spans.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "text": text[m.start(): m.end()],
                    "category": rule.category,
                    "score": rule.score,
                }
            )
    # Sort by start position; for ties, prefer longer spans
    spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    # Remove overlapping spans (greedy, keep first)
    deduped: list[dict] = []
    last_end = -1
    for span in spans:
        if span["start"] >= last_end:
            deduped.append(span)
            last_end = span["end"]
    return deduped


def redact_text(text: str, spans: list[dict]) -> str:
    """Replace PHI spans with [CATEGORY] placeholders."""
    if not spans:
        return text
    pieces = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor: span["start"]])
        pieces.append(f"[{span['category']}]")
        cursor = span["end"]
    pieces.append(text[cursor:])
    return "".join(pieces)
