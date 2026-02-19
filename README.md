# Clinical NLP Pipeline: PHI De-identification + Medical NER

Production-style masters-level project for clinical note processing:
- Hybrid NER (`BiLSTM-CRF` + transformer token classifier) for `diagnosis`, `medication`, `dosage`, `procedure`
- PHI de-identification for 18 HIPAA Safe Harbor categories (hybrid rule + model-ready interface)
- Clinical preprocessing: abbreviation expansion (5,000+ dictionary support), section segmentation, sentence splitting
- FastAPI + Docker deployment with batch processing
- Synthetic large-scale data generation and benchmark-compatible adapters

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
