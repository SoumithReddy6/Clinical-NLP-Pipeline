from __future__ import annotations

import re

LABELS = ["O", "DIAGNOSIS", "MEDICATION", "DOSAGE", "PROCEDURE"]

DOSAGE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b", re.IGNORECASE)

DIAGNOSIS = {"diabetes", "hypertension", "asthma", "pneumonia", "sepsis", "copd"}
MEDICATION = {"metformin", "lisinopril", "atorvastatin", "albuterol", "insulin", "aspirin"}
PROCEDURE = {"mri", "ct", "x-ray", "colonoscopy", "appendectomy", "biopsy"}


def token_level_labels(tokens: list[str]) -> list[str]:
    labels = []
    for tok in tokens:
        t = tok.lower()
        if t in DIAGNOSIS:
            labels.append("DIAGNOSIS")
        elif t in MEDICATION:
            labels.append("MEDICATION")
        elif t in PROCEDURE:
            labels.append("PROCEDURE")
        elif DOSAGE_PATTERN.match(tok):
            labels.append("DOSAGE")
        else:
            labels.append("O")
    return labels
