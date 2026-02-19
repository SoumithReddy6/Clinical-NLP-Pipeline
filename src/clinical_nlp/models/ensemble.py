from __future__ import annotations

from clinical_nlp.schemas import Entity


def merge_entities(model_entities: list[Entity], transformer_entities: list[Entity], threshold: float = 0.6) -> list[Entity]:
    merged = model_entities + [e for e in transformer_entities if e.score >= threshold]
    merged.sort(key=lambda e: (e.start, -(e.end - e.start)))
    output: list[Entity] = []
    last_end = -1
    for entity in merged:
        if entity.start >= last_end:
            output.append(entity)
            last_end = entity.end
    return output
