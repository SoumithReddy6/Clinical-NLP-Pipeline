.PHONY: test run-api run-ui generate-data train-bilstm

test:
	pytest -q

run-api:
	uvicorn clinical_nlp.api.main:app --host 0.0.0.0 --port 8000

run-ui:
	streamlit run streamlit_app.py

generate-data:
	python scripts/generate_synthetic.py --num-notes 10000 --out data/synthetic/clinical_notes_10k.jsonl

train-bilstm:
	python scripts/train_bilstm_crf.py --train data/synthetic/clinical_notes_10k.jsonl --epochs 3 --save-dir artifacts/bilstm_crf
