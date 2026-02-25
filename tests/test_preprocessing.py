from clinical_nlp.preprocessing.pipeline import ClinicalPreprocessor
from clinical_nlp.preprocessing.segmenter import split_sections, split_sentences, is_negated


def test_abbreviation_expansion():
    p = ClinicalPreprocessor()
    out = p.run("Pt has htn and dm.")
    assert "hypertension" in out["normalized_text"].lower()


def test_section_splitting():
    text = (
        "Chief Complaint: headache and fever\n"
        "History of Present Illness: 3-day history of symptoms\n"
        "Medications: ibuprofen 400 mg\n"
        "Assessment and Plan: migraine, continue current meds"
    )
    sections = split_sections(text)
    assert len(sections) == 4
    assert "Chief Complaint" in sections[0]
    assert "Medications" in sections[2]


def test_section_splitting_no_headers():
    text = "This is a plain clinical note without section headers."
    sections = split_sections(text)
    assert len(sections) == 1


def test_sentence_splitting():
    text = "Patient has fever. Started antibiotics. Follow up in 2 weeks."
    sentences = split_sentences(text)
    assert len(sentences) >= 2


def test_preprocessor_returns_all_keys():
    p = ClinicalPreprocessor()
    out = p.run("Chief Complaint: headache")
    assert "normalized_text" in out
    assert "sections" in out
    assert "sentences" in out


def test_negation_detected():
    text = "Patient denies chest pain and shortness of breath."
    assert is_negated(text, text.index("chest"))


def test_negation_not_detected_when_absent():
    text = "Patient reports chest pain and shortness of breath."
    assert not is_negated(text, text.index("chest"))


def test_negation_no_evidence_of():
    text = "No evidence of pulmonary embolism on CT scan."
    assert is_negated(text, text.index("pulmonary"))


def test_negation_window_respects_distance():
    text = "Patient denies headache. " + ("x " * 30) + "Patient has diabetes."
    diabetes_pos = text.index("diabetes")
    assert not is_negated(text, diabetes_pos)


def test_expanded_section_headers():
    text = (
        "Social History: nonsmoker\n"
        "Family History: mother with breast cancer\n"
        "Review of Systems: negative for fever\n"
        "Physical Examination: alert and oriented"
    )
    sections = split_sections(text)
    assert len(sections) == 4
