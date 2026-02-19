from fastapi.testclient import TestClient
from clinical_nlp.api.main import app


def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_extract():
    c = TestClient(app)
    r = c.post("/extract", json={"text": "Patient John Doe has diabetes and takes metformin 500 mg."})
    assert r.status_code == 200
    body = r.json()
    assert "entities" in body
    assert "phi" in body
