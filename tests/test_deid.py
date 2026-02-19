from clinical_nlp.deid.detector import detect_phi_spans, redact_text


def test_detect_and_redact_phi():
    text = "Patient John Doe called from 555-123-4567 on 01/01/2024."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "PHONE" for s in spans)
    assert any(s["category"] == "DATES" for s in spans)
    redacted = redact_text(text, spans)
    assert "[PHONE]" in redacted
