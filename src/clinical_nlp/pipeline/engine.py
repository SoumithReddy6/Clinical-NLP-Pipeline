from __future__ import annotations

from collections import Counter
from typing import Optional

from clinical_nlp.schemas import Entity, PHISpan, ProcessResponse
from clinical_nlp.preprocessing.pipeline import ClinicalPreprocessor
from clinical_nlp.deid.detector import detect_phi_spans, redact_text
from clinical_nlp.models.transformer_ner import TransformerNER, HeuristicClinicalNER
from clinical_nlp.models.ensemble import merge_entities
from clinical_nlp.preprocessing.segmenter import is_negated


class ClinicalNLPPipeline:
    """End-to-end clinical NLP pipeline: preprocessing -> deid -> NER -> ensemble.

    Orchestrates:
    1. Text normalization (abbreviation expansion)
    2. Section segmentation
    3. PHI detection and redaction
    4. Entity extraction (transformer + heuristic ensemble)
    5. Negation detection on extracted entities
    6. Summary statistics
    """

    def __init__(
        self,
        abbr_path: Optional[str] = None,
        transformer_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    ):
        self.preprocessor = ClinicalPreprocessor(abbr_path=abbr_path)
        self.transformer = TransformerNER(model_name=transformer_model)
        self.heuristic = HeuristicClinicalNER()

    def _extract_entities(self, text: str) -> list[Entity]:
        """Run both NER paths and merge via ensemble."""
        base = [
            Entity(
                start=p.start, end=p.end, text=p.text,
                label=p.label, score=p.score, source="transformer",
            )
            for p in self.transformer.predict(text)
            if p.label in {"DIAGNOSIS", "MEDICATION", "DOSAGE", "PROCEDURE"}
        ]
        fallback = [
            Entity(
                start=p.start, end=p.end, text=p.text,
                label=p.label, score=p.score, source="heuristic",
            )
            for p in self.heuristic.predict(text)
        ]
        return merge_entities(base, fallback)

    def _entity_summary(self, entities: list[Entity], text: str) -> dict:
        """Build summary statistics for extracted entities."""
        counts = Counter(e.label for e in entities)
        negated_count = sum(1 for e in entities if is_negated(text, e.start))
        return {
            "total": len(entities),
            "by_type": dict(counts),
            "negated": negated_count,
            "sources": dict(Counter(e.source for e in entities)),
        }

    def process(self, text: str, redact: bool = True) -> ProcessResponse:
        """Run the full pipeline on a single clinical note."""
        pre = self.preprocessor.run(text)
        normalized = pre["normalized_text"]

        phi = detect_phi_spans(normalized)
        redacted = redact_text(normalized, phi) if redact else None

        entities = self._extract_entities(normalized)

        summary = self._entity_summary(entities, normalized)

        return ProcessResponse(
            normalized_text=normalized,
            redacted_text=redacted,
            entities=entities,
            phi=[PHISpan(**p) for p in phi],
            sections=pre["sections"],
            entity_summary=summary,
        )
