from clinical_nlp.preprocessing.pipeline import ClinicalPreprocessor


def test_abbreviation_expansion():
    p = ClinicalPreprocessor()
    out = p.run("Pt has htn and dm.")
    assert "hypertension" in out["normalized_text"]
    assert "diabetes mellitus" in out["normalized_text"]
