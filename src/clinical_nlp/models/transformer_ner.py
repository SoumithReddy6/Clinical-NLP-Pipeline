from __future__ import annotations

import re
from dataclasses import dataclass

from clinical_nlp.models.tokenization import tokenize_with_offsets

try:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except Exception:
    torch = None
    AutoModelForTokenClassification = None
    AutoTokenizer = None


# Maps BIO tags back to base entity labels
_BIO_TO_BASE = {
    "B-DIAGNOSIS": "DIAGNOSIS", "I-DIAGNOSIS": "DIAGNOSIS",
    "B-MEDICATION": "MEDICATION", "I-MEDICATION": "MEDICATION",
    "B-DOSAGE": "DOSAGE", "I-DOSAGE": "DOSAGE",
    "B-PROCEDURE": "PROCEDURE", "I-PROCEDURE": "PROCEDURE",
}

VALID_LABELS = {"DIAGNOSIS", "MEDICATION", "DOSAGE", "PROCEDURE"}


@dataclass
class TransformerPrediction:
    start: int
    end: int
    text: str
    label: str
    score: float


class TransformerNER:
    """Transformer-based NER using fine-tuned token classification models.

    Supports both BIO-tagged and flat-label model outputs.
    Falls back gracefully when no local model is available.
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.available = False
        self.tokenizer = None
        self.model = None
        if AutoTokenizer is not None and AutoModelForTokenClassification is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True)
                self.available = True
            except Exception:
                self.available = False

    def predict(self, text: str) -> list[TransformerPrediction]:
        if not self.available or self.model is None or self.tokenizer is None or torch is None:
            return []
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            out = self.model(**encoded)
            probs = out.logits.softmax(-1)[0]
            pred_ids = probs.argmax(-1).tolist()

        id2label = self.model.config.id2label

        # Merge BIO-tagged subword tokens into contiguous spans
        preds: list[TransformerPrediction] = []
        current_label = None
        current_start = 0
        current_end = 0
        current_scores: list[float] = []

        for i, label_id in enumerate(pred_ids):
            raw_label = id2label.get(label_id, "O")
            s, e = offsets[i]
            if e <= s:
                continue

            base_label = _BIO_TO_BASE.get(raw_label, raw_label if raw_label in VALID_LABELS else None)
            is_begin = raw_label.startswith("B-")
            is_inside = raw_label.startswith("I-")

            if base_label and (is_begin or (not is_inside and base_label in VALID_LABELS)):
                # Flush previous
                if current_label and current_scores:
                    preds.append(TransformerPrediction(
                        start=current_start, end=current_end,
                        text=text[current_start:current_end],
                        label=current_label,
                        score=sum(current_scores) / len(current_scores),
                    ))
                current_label = base_label
                current_start = s
                current_end = e
                current_scores = [float(probs[i][label_id].item())]
            elif is_inside and base_label == current_label:
                # Continue current entity
                current_end = e
                current_scores.append(float(probs[i][label_id].item()))
            else:
                # Flush on O or label change
                if current_label and current_scores:
                    preds.append(TransformerPrediction(
                        start=current_start, end=current_end,
                        text=text[current_start:current_end],
                        label=current_label,
                        score=sum(current_scores) / len(current_scores),
                    ))
                current_label = None
                current_scores = []

        # Flush last
        if current_label and current_scores:
            preds.append(TransformerPrediction(
                start=current_start, end=current_end,
                text=text[current_start:current_end],
                label=current_label,
                score=sum(current_scores) / len(current_scores),
            ))

        return preds


class HeuristicClinicalNER:
    """Dictionary-based clinical NER with massively expanded vocabularies.

    Handles both single-word and multi-word entity recognition.
    """
    DIAGNOSES: set[str] = {
        "diabetes", "hypertension", "hyperlipidemia", "asthma", "pneumonia",
        "sepsis", "copd", "bronchitis", "cirrhosis", "pancreatitis",
        "cholecystitis", "diverticulitis", "appendicitis", "hepatitis",
        "colitis", "gastritis", "stroke", "epilepsy", "migraine",
        "meningitis", "encephalitis", "neuropathy", "dementia",
        "anemia", "lymphoma", "leukemia", "melanoma", "thrombocytopenia",
        "cellulitis", "osteomyelitis", "tuberculosis", "influenza",
        "osteoarthritis", "fibromyalgia", "schizophrenia", "depression",
        "anxiety", "insomnia", "obesity", "gout", "osteoporosis",
        "cardiomyopathy", "endocarditis", "pericarditis", "arrhythmia",
        "hypothyroidism", "hyperthyroidism", "nephrolithiasis",
        "glomerulonephritis",
    }

    DIAGNOSES_MULTI: set[str] = {
        "heart failure", "atrial fibrillation", "coronary artery disease",
        "pulmonary embolism", "deep vein thrombosis", "type 2 diabetes mellitus",
        "type 1 diabetes mellitus", "chronic obstructive pulmonary disease",
        "chronic kidney disease", "acute kidney injury", "urinary tract infection",
        "iron deficiency anemia", "major depressive disorder",
        "gastroesophageal reflux disease", "multiple sclerosis",
        "rheumatoid arthritis", "systemic lupus erythematosus",
        "breast cancer", "prostate cancer", "lung cancer", "colon cancer",
        "multiple myeloma", "bipolar disorder", "traumatic brain injury",
        "alcohol use disorder", "substance use disorder",
    }

    MEDS: set[str] = {
        "metformin", "lisinopril", "atorvastatin", "albuterol", "insulin",
        "aspirin", "amlodipine", "metoprolol", "losartan", "furosemide",
        "omeprazole", "pantoprazole", "acetaminophen", "ibuprofen",
        "gabapentin", "sertraline", "prednisone", "warfarin", "heparin",
        "vancomycin", "ceftriaxone", "azithromycin", "doxycycline",
        "ciprofloxacin", "amoxicillin", "ondansetron", "morphine",
        "fentanyl", "hydromorphone", "oxycodone", "lorazepam",
        "haloperidol", "quetiapine", "levothyroxine", "clopidogrel",
        "apixaban", "enoxaparin", "spironolactone", "carvedilol",
        "diltiazem", "famotidine", "escitalopram", "duloxetine",
        "pregabalin", "fluoxetine", "meropenem", "dexamethasone",
        "methylprednisolone", "epinephrine", "norepinephrine",
    }

    PROCEDURES: set[str] = {
        "mri", "ct", "x-ray", "colonoscopy", "appendectomy", "biopsy",
        "cholecystectomy", "bronchoscopy", "thoracentesis", "paracentesis",
        "tracheostomy", "intubation", "craniotomy", "arthroscopy",
        "mammography", "hemodialysis", "eeg", "ercp",
        "esophagogastroduodenoscopy",
    }

    PROCEDURES_MULTI: set[str] = {
        "chest x-ray", "ct scan", "pet scan", "dexa scan",
        "cardiac catheterization", "coronary angiography",
        "lumbar puncture", "stress test", "bone marrow biopsy",
        "skin biopsy", "liver biopsy", "blood transfusion",
        "spinal fusion", "total hip arthroplasty", "total knee arthroplasty",
    }

    DOSAGE_PATTERN = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|meq|%)\b", re.IGNORECASE
    )

    def predict(self, text: str) -> list[TransformerPrediction]:
        spans: list[TransformerPrediction] = []
        text_lower = text.lower()

        # Multi-word entity matching
        for phrase_set, label, score in [
            (self.DIAGNOSES_MULTI, "DIAGNOSIS", 0.88),
            (self.PROCEDURES_MULTI, "PROCEDURE", 0.85),
        ]:
            for phrase in sorted(phrase_set, key=len, reverse=True):
                start = 0
                pl = phrase.lower()
                while True:
                    idx = text_lower.find(pl, start)
                    if idx < 0:
                        break
                    spans.append(TransformerPrediction(
                        idx, idx + len(phrase), text[idx:idx + len(phrase)], label, score
                    ))
                    start = idx + 1

        # Single-word matching
        for tok, s, e in tokenize_with_offsets(text):
            t = tok.lower()
            if t in self.DIAGNOSES:
                spans.append(TransformerPrediction(s, e, tok, "DIAGNOSIS", 0.90))
            elif t in self.MEDS:
                spans.append(TransformerPrediction(s, e, tok, "MEDICATION", 0.90))
            elif t in self.PROCEDURES:
                spans.append(TransformerPrediction(s, e, tok, "PROCEDURE", 0.88))

        # Dosage pattern matching
        for m in self.DOSAGE_PATTERN.finditer(text):
            spans.append(TransformerPrediction(
                m.start(), m.end(), m.group(), "DOSAGE", 0.92
            ))

        # Deduplicate: sort by start, prefer longer spans
        spans.sort(key=lambda x: (x.start, -(x.end - x.start)))
        deduped: list[TransformerPrediction] = []
        last_end = -1
        for sp in spans:
            if sp.start >= last_end:
                deduped.append(sp)
                last_end = sp.end

        return deduped
