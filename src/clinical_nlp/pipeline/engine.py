from __future__ import annotations

from typing import Optional

from clinical_nlp.schemas import Entity, PHISpan, ProcessResponse
from clinical_nlp.preprocessing.pipeline import ClinicalPreprocessor
from clinical_nlp.deid.detector import detect_phi_spans, redact_text
from clinical_nlp.models.transformer_ner import TransformerNER, HeuristicClinicalNER
from clinical_nlp.models.ensemble import merge_entities


class ClinicalNLPPipeline:
    def __init__(self, abbr_path: Optional[str] = None, transformer_model: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.preprocessor = ClinicalPreprocessor(abbr_path=abbr_path)
        self.transformer = TransformerNER(model_name=transformer_model)
        self.heuristic = HeuristicClinicalNER()

    def _extract_entities(self, text: str) -> list[Entity]:
        base = [
            Entity(start=p.start, end=p.end, text=p.text, label=p.label, score=p.score, source="transformer")
            for p in self.transformer.predict(text)
            if p.label in {"DIAGNOSIS", "MEDICATION", "DOSAGE", "PROCEDURE"}
        ]
        fallback = [
            Entity(start=p.start, end=p.end, text=p.text, label=p.label, score=p.score, source="heuristic")
            for p in self.heuristic.predict(text)
        ]
        return merge_entities(base, fallback)

    def process(self, text: str, redact: bool = True) -> ProcessResponse:
        pre = self.preprocessor.run(text)
        phi = detect_phi_spans(pre["normalized_text"])
        redacted = redact_text(pre["normalized_text"], phi) if redact else None
        entities = self._extract_entities(pre["normalized_text"])
        return ProcessResponse(
            normalized_text=pre["normalized_text"],
            redacted_text=redacted,
            entities=entities,
            phi=[PHISpan(**p) for p in phi],
            sections=pre["sections"],
        )
