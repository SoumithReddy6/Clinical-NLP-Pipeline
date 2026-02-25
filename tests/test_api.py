from fastapi.testclient import TestClient
from clinical_nlp.api.main import app


def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_extract():
    c = TestClient(app)
    r = c.post("/extract", json={
        "text": "Patient John Doe has diabetes and takes metformin 500 mg. Phone: 555-123-4567"
    })
    assert r.status_code == 200
    body = r.json()
    assert "entities" in body
    assert "phi" in body
    assert "entity_summary" in body
    assert len(body["entities"]) > 0
    labels = {e["label"] for e in body["entities"]}
    assert "DIAGNOSIS" in labels or "MEDICATION" in labels


def test_extract_with_redaction():
    c = TestClient(app)
    r = c.post("/extract", json={
        "text": "Patient Jane Smith has hypertension. SSN: 123-45-6789",
        "redact": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["redacted_text"] is not None
    assert "123-45-6789" not in body["redacted_text"]


def test_extract_no_redaction():
    c = TestClient(app)
    r = c.post("/extract", json={
        "text": "Patient has pneumonia.",
        "redact": False,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["redacted_text"] is None


def test_extract_entity_summary():
    c = TestClient(app)
    r = c.post("/extract", json={
        "text": "Patient has diabetes and hypertension. Takes metformin 500 mg daily."
    })
    body = r.json()
    summary = body.get("entity_summary")
    assert summary is not None
    assert "total" in summary
    assert "by_type" in summary
    assert summary["total"] > 0


def test_batch_extract():
    c = TestClient(app)
    r = c.post("/extract/batch", json={
        "documents": [
            "Patient has diabetes and takes metformin 500 mg.",
            "Patient Jane Doe with pneumonia and hypertension. Phone: 555-111-2222",
        ],
        "redact": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert body["total_entities"] > 0
    assert body["total_phi_spans"] >= 0


def test_batch_extract_totals():
    c = TestClient(app)
    r = c.post("/extract/batch", json={
        "documents": [
            "Patient has sepsis. Takes vancomycin 1000 mg.",
            "Coronary artery disease diagnosed. Aspirin 81 mg daily.",
        ],
    })
    body = r.json()
    entity_sum = sum(len(res["entities"]) for res in body["results"])
    assert body["total_entities"] == entity_sum
    phi_sum = sum(len(res["phi"]) for res in body["results"])
    assert body["total_phi_spans"] == phi_sum
