# Clinical NLP Pipeline for Medical Record De-identification & Entity Extraction

## [Live Demo](https://clinical-nlp-pipeline-aoy4w3niqdwxn3f3jpgekw.streamlit.app) 🚀

An end-to-end clinical Natural Language Processing pipeline that performs **HIPAA-compliant PHI de-identification** (all 18 Safe Harbor categories) and **medical Named Entity Recognition** (NER) on clinical notes. Built with PyTorch, HuggingFace Transformers, FastAPI, and Streamlit.

## Features

### De-identification (PHI Detection)
- Coverage of **all 18 HIPAA Safe Harbor categories**: Names, Geographic Subdivisions, Dates, Phone Numbers, Fax Numbers, Email Addresses, SSNs, MRNs, Health Plan Beneficiary Numbers, Account Numbers, Certificate/License Numbers, Vehicle Identifiers, Device Identifiers, URLs, IP Addresses, Biometric Identifiers, Full-Face Photos, and Other Unique Identifiers
- Regex-based detection with confidence scoring
- Greedy non-overlapping span resolution
- Configurable redaction with category-specific placeholders (e.g., `[NAME]`, `[SSN]`, `[PHONE]`)

### Named Entity Recognition
- **4 entity types**: Diagnosis, Medication, Dosage, Procedure
- **BIO tagging scheme** (9 labels): O, B-DIAGNOSIS, I-DIAGNOSIS, B-MEDICATION, I-MEDICATION, B-DOSAGE, I-DOSAGE, B-PROCEDURE, I-PROCEDURE
- **Multi-word entity support**: "chronic obstructive pulmonary disease", "insulin glargine", "total hip arthroplasty"
- **Three NER models**:
  - **Transformer NER**: Fine-tunable Bio_ClinicalBERT with BIO tag merging
  - **BiLSTM-CRF**: Bidirectional LSTM with a real Conditional Random Field layer (forward algorithm + Viterbi decoding)
  - **Heuristic NER**: Dictionary-based with 200+ medical terms and dosage pattern matching
- **Ensemble merging**: Score-based greedy non-overlapping selection with source boosting

### Clinical Negation Detection
- NegEx-style negation cue detection (denies, no evidence of, without, absent, etc.)
- 60-character lookback window for negation scope
- Negated entities flagged in output and crossed out in the UI

### Preprocessing
- **600+ real clinical abbreviations** (vital signs, lab terms, dosing frequencies, anatomical terms, ICU terms, etc.)
- **25+ section header patterns** (Chief Complaint, HPI, Social History, Review of Systems, Physical Exam, Discharge Summary, etc.)
- Sentence segmentation

### Evaluation
- Per-entity-type precision, recall, and F1 scores
- Micro and macro averages
- Optional relaxed (overlap-based) matching
- Progress reporting for large datasets

## Project Structure

```
.
├── src/clinical_nlp/
│   ├── api/main.py                  # FastAPI REST endpoints
│   ├── data/
│   │   ├── synthetic.py             # Synthetic clinical note generator
│   │   └── io.py                    # JSONL read/write utilities
│   ├── deid/
│   │   ├── detector.py              # PHI detection (18 HIPAA categories)
│   │   └── categories.py            # HIPAA category list
│   ├── models/
│   │   ├── bilstm_crf.py            # BiLSTM-CRF with real CRF layer
│   │   ├── transformer_ner.py       # Transformer + Heuristic NER
│   │   ├── ensemble.py              # Entity merging strategy
│   │   └── tokenization.py          # Offset-preserving tokenizer
│   ├── pipeline/engine.py           # Main orchestration pipeline
│   ├── preprocessing/
│   │   ├── abbreviations.py         # Abbreviation expansion
│   │   ├── segmenter.py             # Section/sentence splitting + negation
│   │   └── pipeline.py              # Preprocessing orchestrator
│   ├── schemas.py                   # Pydantic request/response models
│   └── training/labeling.py         # BIO token-level labeling
├── scripts/
│   ├── generate_synthetic.py        # Generate synthetic datasets
│   ├── train_bilstm_crf.py          # Train BiLSTM-CRF model
│   ├── train_transformer_ner.py     # Fine-tune transformer NER
│   └── evaluate_pipeline.py         # Evaluate pipeline performance
├── data/resources/
│   └── abbreviations_5200.tsv       # 600+ real clinical abbreviations
├── configs/default.yaml             # Pipeline configuration
├── tests/                           # Unit tests
├── streamlit_app.py                 # Interactive Streamlit demo
├── Makefile                         # Build targets
└── pyproject.toml                   # Package configuration
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Clinical NLP Pipeline for Medical Record De-identification & Entity Extraction"

# Install dependencies
pip install -e ".[dev]"
```

### Requirements
- Python >= 3.9
- PyTorch >= 2.2
- HuggingFace Transformers >= 4.44
- FastAPI, Streamlit, Pydantic, seqeval

## Quick Start

### Streamlit Demo
```bash
make run-ui
# or
streamlit run streamlit_app.py
```

### REST API
```bash
make run-api
# or
uvicorn clinical_nlp.api.main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `GET /health` — Health check
- `POST /extract` — Process a single clinical note
- `POST /extract/batch` — Process multiple notes

**Example request:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient John Doe has diabetes and takes metformin 500 mg daily. SSN: 123-45-6789", "redact": true}'
```

### Python API
```python
from clinical_nlp.pipeline.engine import ClinicalNLPPipeline

pipeline = ClinicalNLPPipeline()
result = pipeline.process(
    "Patient John Doe has type 2 diabetes mellitus and takes metformin 500 mg daily.",
    redact=True,
)

print(f"Entities: {[(e.text, e.label) for e in result.entities]}")
print(f"PHI: {[(p.text, p.category) for p in result.phi]}")
print(f"Redacted: {result.redacted_text}")
print(f"Summary: {result.entity_summary}")
```

## Training

### Generate Synthetic Data
```bash
make generate-data
# or
python scripts/generate_synthetic.py --num-notes 10000 --out data/synthetic/clinical_notes_10k.jsonl
```

The synthetic data generator produces realistic clinical notes with:
- 100+ diagnoses, 130+ medications, 60+ dosage patterns, 70+ procedures
- Structured templates with Chief Complaint, HPI, Allergies, Medications, Vitals, Physical Exam, Labs, Assessment and Plan
- Accurate character-offset entity annotations

### Train BiLSTM-CRF
```bash
make train-bilstm
# or
python scripts/train_bilstm_crf.py --train data/synthetic/clinical_notes_10k.jsonl --epochs 5 --save-dir artifacts/bilstm_crf
```

Uses AdamW optimizer with OneCycleLR scheduler and CRF negative log-likelihood loss.

### Fine-tune Transformer NER
```bash
make train-transformer
# or
python scripts/train_transformer_ner.py --train data/synthetic/clinical_notes_10k.jsonl --epochs 3 --save-dir artifacts/transformer_ner
```

Fine-tunes Bio_ClinicalBERT with train/eval split, warmup, and per-entity-type metrics.

### Evaluate
```bash
make evaluate
# or
python scripts/evaluate_pipeline.py --input data/synthetic/clinical_notes_1k.jsonl --num 200
```

## Testing

```bash
make test
# or
pytest -q
```

Test coverage includes:
- **De-identification**: All 18 PHI categories, redaction, overlap resolution
- **Preprocessing**: Abbreviation expansion, section splitting, negation detection
- **API**: Health check, single/batch extraction, entity summaries, redaction toggle

## Pipeline Architecture

```
Clinical Note
    │
    ▼
┌─────────────────────┐
│   Preprocessing      │
│  • Abbreviation exp. │
│  • Section splitting │
│  • Sentence splitting│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  De-identification   │
│  • 18 HIPAA regex    │
│  • Span dedup        │
│  • Text redaction    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Entity Extraction   │
│  • Transformer NER   │
│  • Heuristic NER     │
│  • Ensemble merge    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Post-processing     │
│  • Negation detect   │
│  • Entity summary    │
│  • JSON response     │
└─────────────────────┘
```

## Configuration

See `configs/default.yaml` for all configurable parameters:
- NER labels (BIO scheme)
- HIPAA categories
- Training hyperparameters (BiLSTM and Transformer)
- Evaluation settings
- Preprocessing options

## Technologies

- **PyTorch** — BiLSTM-CRF model, CRF layer with Viterbi decoding
- **HuggingFace Transformers** — Bio_ClinicalBERT fine-tuning
- **FastAPI** — REST API with Pydantic validation
- **Streamlit** — Interactive demo with entity highlighting
- **Pydantic** — Request/response schemas
- **seqeval** — Sequence labeling evaluation metrics
