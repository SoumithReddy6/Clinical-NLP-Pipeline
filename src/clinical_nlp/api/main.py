from __future__ import annotations

from fastapi import FastAPI

from clinical_nlp.schemas import (
    BatchProcessRequest,
    BatchProcessResponse,
    ProcessRequest,
    ProcessResponse,
)
from clinical_nlp.pipeline.engine import ClinicalNLPPipeline

app = FastAPI(title="Clinical NLP Pipeline", version="0.2.0")
pipeline = ClinicalNLPPipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract", response_model=ProcessResponse)
def extract(req: ProcessRequest) -> ProcessResponse:
    return pipeline.process(req.text, redact=req.redact)


@app.post("/extract/batch", response_model=BatchProcessResponse)
def extract_batch(req: BatchProcessRequest) -> BatchProcessResponse:
    results = [pipeline.process(doc, redact=req.redact) for doc in req.documents]
    total_entities = sum(len(r.entities) for r in results)
    total_phi_spans = sum(len(r.phi) for r in results)
    return BatchProcessResponse(
        results=results,
        total_entities=total_entities,
        total_phi_spans=total_phi_spans,
    )
