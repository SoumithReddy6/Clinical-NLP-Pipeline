# Clinical NLP Pipeline: PHI De-identification + Medical NER
## [Live Demo]([https://YOUR-APP-URL.streamlit.app](https://clinical-nlp-pipeline-aoy4w3niqdwxn3f3jpgekw.streamlit.app) 🚀


## Project Description

This project is an end-to-end **Clinical NLP pipeline** for processing unstructured medical notes.  
It focuses on two core tasks:

1. **PHI De-identification**: Detects and redacts protected health information (PHI) across HIPAA-relevant categories.
2. **Medical Entity Extraction**: Identifies key clinical entities such as **diagnoses, medications, dosages, and procedures**.

The system combines rule-based clinical text processing (abbreviation expansion, section segmentation, sentence splitting) with a hybrid NER approach using **BiLSTM-CRF** and a **transformer-based model**. It is exposed through both a **FastAPI REST API** and a **Streamlit web app** for interactive demos and batch workflows.

### Key Features
- Clinical note preprocessing pipeline
- PHI detection + redaction module (18-category framework)
- Hybrid NER architecture (BiLSTM-CRF + transformer + ensemble merge)
- Batch and single-document inference
- FastAPI deployment and Streamlit live demo
- Synthetic large-scale dataset generator and benchmark-style data adapters

> Note: This repository uses **synthetic clinical data by default** for safe public demonstration and development.


## Architecture

1. `preprocessing`: normalizes text and sections clinical notes
2. `deid`: detects PHI spans and redacts text
3. `models`: runs BiLSTM-CRF and transformer NER
4. `pipeline`: merges outputs and returns structured JSON
5. `api`: exposes `/extract` and `/extract/batch`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate large synthetic dataset

```bash
python scripts/generate_synthetic.py \
  --num-notes 100000 \
  --out data/synthetic/clinical_notes_100k.jsonl
```

### Train BiLSTM-CRF baseline

```bash
python scripts/train_bilstm_crf.py \
  --train data/synthetic/clinical_notes_100k.jsonl \
  --max-samples 20000 \
  --batch-size 256 \
  --epochs 5 \
  --save-dir artifacts/bilstm_crf
```

### Fine-tune transformer model

```bash
python scripts/train_transformer_ner.py \
  --train data/synthetic/clinical_notes_100k.jsonl \
  --model emilyalsentzer/Bio_ClinicalBERT \
  --save-dir artifacts/clinicalbert_ner
```

### Evaluate

```bash
python scripts/evaluate_pipeline.py \
  --data data/synthetic/clinical_notes_100k.jsonl \
  --limit 2000
```

### Run API

```bash
uvicorn clinical_nlp.api.main:app --host 0.0.0.0 --port 8000
```

### Run Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

Then open: `http://localhost:8501`

Example request:

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"Patient John Doe with Type 2 diabetes started metformin 500 mg daily on 01/03/2024."}'
```

## Docker

```bash
docker build -t clinical-nlp-pipeline .
docker run -p 8000:8000 clinical-nlp-pipeline
```

## Lightweight website deployment options

- Streamlit Community Cloud: deploy directly from your GitHub repo (`streamlit_app.py` as entrypoint)
- Render/Railway: run with start command `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

## Benchmarks and real datasets

- Includes schema compatible with i2b2/n2c2-style token span annotations
- For restricted datasets (MIMIC, i2b2, n2c2), use the adapters in `scripts/` and place source files under `data/raw/`

## Project Layout

```text
src/clinical_nlp/
  api/
  data/
  deid/
  models/
  pipeline/
  preprocessing/
  training/
scripts/
tests/
```

## Notes

- This repo ships synthetic data tooling; real clinical corpora often require credentialed access and DUA approval.
- Default configs are optimized for reproducibility and demonstration. For production, calibrate thresholds on held-out site-specific data.
