from __future__ import annotations

from pathlib import Path
from typing import Optional


DEFAULT_MAP = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "cad": "coronary artery disease",
    "copd": "chronic obstructive pulmonary disease",
    "sob": "shortness of breath",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
}


def load_abbreviation_map(path: Optional[str] = None) -> dict[str, str]:
    if not path:
        return DEFAULT_MAP.copy()
    mapping = DEFAULT_MAP.copy()
    file = Path(path)
    if not file.exists():
        return mapping
    with file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                continue
            short, long = line.split("\t", 1)
            mapping[short.strip().lower()] = long.strip().lower()
    return mapping


def expand_abbreviations(text: str, mapping: dict[str, str]) -> str:
    tokens = text.split()
    expanded = []
    for tok in tokens:
        clean = tok.lower().strip(".,;:!?()[]{}")
        replacement = mapping.get(clean)
        expanded.append(replacement if replacement else tok)
    return " ".join(expanded)
