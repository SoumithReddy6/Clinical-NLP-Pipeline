from __future__ import annotations

from typing import Optional

from clinical_nlp.preprocessing.abbreviations import load_abbreviation_map, expand_abbreviations
from clinical_nlp.preprocessing.segmenter import split_sections, split_sentences, is_negated


class ClinicalPreprocessor:
    def __init__(self, abbr_path: Optional[str] = None):
        self.abbr_map = load_abbreviation_map(abbr_path)

    def run(self, text: str) -> dict:
        normalized = expand_abbreviations(text, self.abbr_map)
        sections = split_sections(normalized)
        sentences = []
        for section in sections:
            sentences.extend(split_sentences(section))
        return {
            "normalized_text": normalized,
            "sections": sections,
            "sentences": sentences,
        }
