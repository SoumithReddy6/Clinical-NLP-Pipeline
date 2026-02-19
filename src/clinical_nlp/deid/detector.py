from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MatchRule:
    category: str
    pattern: re.Pattern[str]
    score: float = 0.95


RULES = [
    MatchRule("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    MatchRule("PHONE", re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    MatchRule("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    MatchRule("DATES", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")),
    MatchRule("URL", re.compile(r"\bhttps?://[^\s]+\b")),
    MatchRule("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    MatchRule("MRN", re.compile(r"\b(?:MRN|Medical Record Number)[:\s#-]*\d{6,12}\b", re.IGNORECASE)),
    MatchRule("ACCOUNT", re.compile(r"\b(?:Acct|Account)[:\s#-]*\d{6,16}\b", re.IGNORECASE)),
    MatchRule("NAME", re.compile(r"\b(?:Patient|Pt)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b"), 0.85),
    MatchRule("GEOGRAPHIC_SUBDIVISION", re.compile(r"\b\d{2,5}\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd)\b"), 0.8),
]


def detect_phi_spans(text: str) -> list[dict]:
    spans = []
    for rule in RULES:
        for m in rule.pattern.finditer(text):
            spans.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "text": text[m.start() : m.end()],
                    "category": rule.category,
                    "score": rule.score,
                }
            )
    spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    deduped = []
    last_end = -1
    for span in spans:
        if span["start"] >= last_end:
            deduped.append(span)
            last_end = span["end"]
    return deduped


def redact_text(text: str, spans: list[dict]) -> str:
    if not spans:
        return text
    pieces = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span["start"]])
        pieces.append(f"[{span['category']}]")
        cursor = span["end"]
    pieces.append(text[cursor:])
    return "".join(pieces)
