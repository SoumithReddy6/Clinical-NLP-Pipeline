from __future__ import annotations

import re

# Extended section headers covering more clinical note formats
SECTION_PATTERN = re.compile(
    r"(?im)^("
    r"chief complaint|history of present illness|past medical history|"
    r"past surgical history|family history|social history|"
    r"review of systems|medications|allergies|"
    r"physical exam(?:ination)?|vital(?:\s+signs?)?|laboratory|labs|"
    r"imaging|radiology|"
    r"assessment(?:\s+and\s+plan)?|plan|"
    r"hospital course|discharge (?:diagnosis|medications|instructions|summary|plan)|"
    r"impression|recommendations?|"
    r"procedures?|operative (?:note|findings)|"
    r"consultations?|"
    r"neurological exam|cardiovascular|respiratory|gastrointestinal|"
    r"musculoskeletal|psychiatric|"
    r"problem list|active problems"
    r"):"
)

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Clinical negation patterns for downstream use
NEGATION_PATTERN = re.compile(
    r"\b("
    r"no|not|none|never|neither|nor|"
    r"without|absent|denies?|denied|"
    r"negative|unremarkable|"
    r"rules?\s+out|ruled\s+out|r/o|"
    r"no\s+evidence\s+of|"
    r"no\s+signs?\s+of|"
    r"does\s+not|did\s+not|cannot|could\s+not|"
    r"unlikely|improbable"
    r")\b",
    re.IGNORECASE,
)

# Window (in characters) to look for negation before an entity
NEGATION_WINDOW = 60


def split_sections(text: str) -> list[str]:
    """Split clinical text into sections based on standard section headers."""
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []
    sections: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            sections.append(chunk)
    return sections


def split_sentences(section_text: str) -> list[str]:
    """Split a section into sentences."""
    sentences = [s.strip() for s in SENTENCE_PATTERN.split(section_text) if s.strip()]
    return sentences


def is_negated(text: str, entity_start: int) -> bool:
    """Check whether an entity at the given position is preceded by a negation cue.

    Looks backwards from entity_start within a window for negation patterns.
    """
    window_start = max(0, entity_start - NEGATION_WINDOW)
    window_text = text[window_start:entity_start]
    return bool(NEGATION_PATTERN.search(window_text))
