from __future__ import annotations

from dataclasses import dataclass

from clinical_nlp.models.tokenization import tokenize_with_offsets

try:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except Exception:
    torch = None
    AutoModelForTokenClassification = None
    AutoTokenizer = None


@dataclass
class TransformerPrediction:
    start: int
    end: int
    text: str
    label: str
    score: float


class TransformerNER:
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
        preds: list[TransformerPrediction] = []
        for i, label_id in enumerate(pred_ids):
            label = id2label.get(label_id, "O")
            if label == "O":
                continue
            s, e = offsets[i]
            if e <= s:
                continue
            score = float(probs[i][label_id].item())
            preds.append(TransformerPrediction(start=s, end=e, text=text[s:e], label=label, score=score))
        return preds


class HeuristicClinicalNER:
    DIAGNOSES = {"diabetes", "hypertension", "asthma", "pneumonia", "sepsis", "copd"}
    MEDS = {"metformin", "lisinopril", "atorvastatin", "albuterol", "insulin", "aspirin"}
    PROCEDURES = {"mri", "ct", "x-ray", "colonoscopy", "appendectomy", "biopsy"}

    def predict(self, text: str) -> list[TransformerPrediction]:
        spans = []
        for tok, s, e in tokenize_with_offsets(text):
            t = tok.lower()
            if t in self.DIAGNOSES:
                spans.append(TransformerPrediction(s, e, tok, "DIAGNOSIS", 0.9))
            elif t in self.MEDS:
                spans.append(TransformerPrediction(s, e, tok, "MEDICATION", 0.9))
            elif t in self.PROCEDURES:
                spans.append(TransformerPrediction(s, e, tok, "PROCEDURE", 0.88))
        return spans
