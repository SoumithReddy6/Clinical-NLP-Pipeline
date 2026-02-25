from __future__ import annotations

from collections import defaultdict
from clinical_nlp.schemas import Entity


def merge_entities(
    primary_entities: list[Entity],
    fallback_entities: list[Entity],
    threshold: float = 0.5,
    primary_boost: float = 0.1,
) -> list[Entity]:
    """Merge entities from primary (transformer) and fallback (heuristic) sources.

    Strategy:
    1. Primary entities above threshold are kept.
    2. Fallback entities fill gaps not covered by primary.
    3. If both sources detect the same span, keep the higher-scoring one.
    4. Deduplication removes overlapping spans (greedy longest match).
    """
    # Collect all candidates with source-aware scoring
    candidates: list[Entity] = []

    for e in primary_entities:
        if e.score >= threshold:
            # Boost primary source slightly for ensemble preference
            candidates.append(e.model_copy(update={"score": min(1.0, e.score + primary_boost)}))

    for e in fallback_entities:
        if e.score >= threshold:
            candidates.append(e)

    if not candidates:
        return []

    # Sort by score descending, then by span length descending (prefer longer)
    candidates.sort(key=lambda e: (-e.score, e.start, -(e.end - e.start)))

    # Greedy non-overlapping selection
    selected: list[Entity] = []
    occupied: list[tuple[int, int]] = []

    for entity in candidates:
        overlaps = any(
            entity.start < occ_end and entity.end > occ_start
            for occ_start, occ_end in occupied
        )
        if not overlaps:
            selected.append(entity)
            occupied.append((entity.start, entity.end))

    # Sort final output by position
    selected.sort(key=lambda e: (e.start, -(e.end - e.start)))
    return selected
