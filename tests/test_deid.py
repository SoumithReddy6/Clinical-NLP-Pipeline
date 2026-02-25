from clinical_nlp.deid.detector import detect_phi_spans, redact_text


def test_detect_phone():
    text = "Call 555-123-4567 for results."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "PHONE" for s in spans)


def test_detect_dates():
    text = "Visit scheduled for 01/15/2024."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "DATES" for s in spans)


def test_detect_date_long_format():
    text = "Admitted on January 15, 2024."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "DATES" for s in spans)


def test_detect_email():
    text = "Contact at john.doe@hospital.com for follow-up."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "EMAIL" for s in spans)


def test_detect_ssn():
    text = "SSN: 123-45-6789 on file."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "SSN" for s in spans)


def test_detect_mrn():
    text = "MRN: 12345678 assigned."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "MRN" for s in spans)


def test_detect_url():
    text = "Records at https://portal.hospital.org/patient"
    spans = detect_phi_spans(text)
    assert any(s["category"] == "URL" for s in spans)


def test_detect_ip():
    text = "Logged from IP 192.168.1.100 today."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "IP_ADDRESS" for s in spans)


def test_detect_name():
    text = "Patient John Smith presented to the ER."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "NAME" for s in spans)


def test_detect_fax():
    text = "Fax: 555-987-6543 for reports."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "FAX" for s in spans)


def test_detect_account():
    text = "Account: 9876543210 active."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "ACCOUNT" for s in spans)


def test_detect_certificate_license():
    text = "DEA: AB1234567 on file."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "CERTIFICATE_LICENSE" for s in spans)


def test_detect_device_identifier():
    text = "Device: SN123456789012 implanted."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "DEVICE_IDENTIFIER" for s in spans)


def test_detect_health_plan():
    text = "Insurance ID: ABC123456789 active."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "HEALTH_PLAN_BENEFICIARY" for s in spans)


def test_detect_patient_id():
    text = "Patient ID: PAT12345678 in system."
    spans = detect_phi_spans(text)
    assert any(s["category"] == "OTHER_UNIQUE_IDENTIFIER" for s in spans)


def test_redact_phi():
    text = "Patient John Doe called from 555-123-4567 on 01/01/2024."
    spans = detect_phi_spans(text)
    redacted = redact_text(text, spans)
    assert "[PHONE]" in redacted
    assert "555-123-4567" not in redacted


def test_redact_preserves_text_without_phi():
    text = "Normal clinical note with no identifiers."
    spans = detect_phi_spans(text)
    redacted = redact_text(text, spans)
    assert redacted == text


def test_overlapping_spans_resolved():
    text = "SSN 123-45-6789 and phone 555-123-4567."
    spans = detect_phi_spans(text)
    for i in range(len(spans) - 1):
        assert spans[i]["end"] <= spans[i + 1]["start"]
