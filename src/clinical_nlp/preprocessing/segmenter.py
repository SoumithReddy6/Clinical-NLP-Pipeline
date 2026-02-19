from __future__ import annotations

import re

SECTION_PATTERN = re.compile(
    r"(?im)^(chief complaint|history of present illness|past medical history|medications|assessment|plan):"
)

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sections(text: str) -> list[str]:
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
    sentences = [s.strip() for s in SENTENCE_PATTERN.split(section_text) if s.strip()]
    return sentences
