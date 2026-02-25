.PHONY: test run-api run-ui generate-data train-bilstm train-transformer evaluate lint

test:
	pytest -q

run-api:
	uvicorn clinical_nlp.api.main:app --host 0.0.0.0 --port 8000

run-ui:
	streamlit run streamlit_app.py

generate-data:
	python scripts/generate_synthetic.py --num-notes 10000 --out data/synthetic/clinical_notes_10k.jsonl

train-bilstm:
	python scripts/train_bilstm_crf.py --train data/synthetic/clinical_notes_10k.jsonl --epochs 5 --save-dir artifacts/bilstm_crf

train-transformer:
	python scripts/train_transformer_ner.py --train data/synthetic/clinical_notes_10k.jsonl --epochs 3 --save-dir artifacts/transformer_ner

evaluate:
	python scripts/evaluate_pipeline.py --input data/synthetic/clinical_notes_1k.jsonl --num 200

lint:
	python -m py_compile src/clinical_nlp/pipeline/engine.py
	python -m py_compile src/clinical_nlp/deid/detector.py
	python -m py_compile src/clinical_nlp/models/bilstm_crf.py
	python -m py_compile src/clinical_nlp/models/transformer_ner.py
	python -m py_compile src/clinical_nlp/training/labeling.py
