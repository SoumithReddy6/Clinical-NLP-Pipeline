#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from clinical_nlp.pipeline.engine import ClinicalNLPPipeline
from clinical_nlp.preprocessing.segmenter import is_negated

# Entity label color map for highlighting
ENTITY_COLORS = {
    "DIAGNOSIS": "#ff6b6b",
    "MEDICATION": "#51cf66",
    "DOSAGE": "#339af0",
    "PROCEDURE": "#fcc419",
}
PHI_COLOR = "#e599f7"


def split_notes(raw: str) -> list[str]:
    chunks = [chunk.strip() for chunk in raw.split("\n---\n")]
    return [chunk for chunk in chunks if chunk]


@st.cache_resource
def get_pipeline() -> ClinicalNLPPipeline:
    return ClinicalNLPPipeline()


def _annotated_text_html(text: str, entities: list, phi_spans: list) -> str:
    """Build HTML with colored entity and PHI highlights.

    Uses inline-block spans so that each annotated entity wraps as a single
    unit regardless of browser width, preventing label/text misalignment.
    """
    spans = []
    for e in entities:
        d = e if isinstance(e, dict) else e.model_dump()
        negated = is_negated(text, d["start"])
        spans.append({
            "start": d["start"],
            "end": d["end"],
            "label": d["label"],
            "color": ENTITY_COLORS.get(d["label"], "#adb5bd"),
            "negated": negated,
        })
    for p in phi_spans:
        d = p if isinstance(p, dict) else p.model_dump()
        spans.append({
            "start": d["start"],
            "end": d["end"],
            "label": d["category"],
            "color": PHI_COLOR,
            "negated": False,
        })

    spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    deduped = []
    last_end = -1
    for s in spans:
        if s["start"] >= last_end:
            deduped.append(s)
            last_end = s["end"]

    parts = []
    cursor = 0
    for s in deduped:
        if s["start"] > cursor:
            parts.append(_escape(text[cursor:s["start"]]))
        neg_style = "text-decoration:line-through;" if s["negated"] else ""
        entity_text = _escape(text[s["start"]:s["end"]])
        label = s["label"]
        color = s["color"]
        # Use box-decoration-break so padding/background works across line wraps
        parts.append(
            f'<mark style="background-color:{color};padding:1px 4px;'
            f'border-radius:3px;color:#222;{neg_style}'
            f'box-decoration-break:clone;-webkit-box-decoration-break:clone;">'
            f'{entity_text}'
            f'<sub style="font-size:0.6em;font-weight:700;margin-left:2px;'
            f'opacity:0.7;">{label}</sub>'
            f'</mark>'
        )
        cursor = s["end"]
    if cursor < len(text):
        parts.append(_escape(text[cursor:]))

    return "".join(parts)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def _render_legend() -> None:
    cols = st.columns(len(ENTITY_COLORS) + 1)
    for i, (label, color) in enumerate(ENTITY_COLORS.items()):
        cols[i].markdown(
            f'<span style="background-color:{color};padding:3px 8px;'
            f'border-radius:4px;font-size:0.85em;font-weight:bold;">'
            f'{label}</span>',
            unsafe_allow_html=True,
        )
    cols[-1].markdown(
        f'<span style="background-color:{PHI_COLOR};padding:3px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:bold;">PHI</span>',
        unsafe_allow_html=True,
    )


def _render_metrics(summary: dict | None, phi_count: int) -> None:
    if not summary:
        return
    by_type = summary.get("by_type", {})
    cols = st.columns(6)
    cols[0].metric("Total Entities", summary.get("total", 0))
    cols[1].metric("Diagnoses", by_type.get("DIAGNOSIS", 0))
    cols[2].metric("Medications", by_type.get("MEDICATION", 0))
    cols[3].metric("Dosages", by_type.get("DOSAGE", 0))
    cols[4].metric("Procedures", by_type.get("PROCEDURE", 0))
    cols[5].metric("PHI Spans", phi_count)

    negated = summary.get("negated", 0)
    if negated > 0:
        st.caption(f"Negated entities: {negated}")

    sources = summary.get("sources", {})
    if sources:
        st.caption(f"Sources: {', '.join(f'{k}={v}' for k, v in sources.items())}")


def render_single_note(pipeline: ClinicalNLPPipeline) -> None:
    st.subheader("Single Note")
    text = st.text_area(
        "Clinical note",
        value=(
            "Chief Complaint: Patient John Doe presents with type 2 diabetes mellitus and hypertension.\n"
            "History of Present Illness: 58-year-old male with a history of coronary artery disease, "
            "started metformin 500 mg daily and lisinopril 10 mg on 01/03/2024. "
            "Patient denies chest pain. No evidence of pulmonary embolism.\n"
            "Allergies: Penicillin\n"
            "Procedures: CT scan of the chest, transthoracic echocardiogram\n"
            "Phone: 555-123-4567 Email: john.doe@email.com SSN: 123-45-6789"
        ),
        height=200,
    )

    col_opt1, col_opt2 = st.columns(2)
    redact = col_opt1.checkbox("Redact PHI", value=True)
    show_highlight = col_opt2.checkbox("Highlight entities", value=True)

    if st.button("Run Extraction", type="primary"):
        if not text.strip():
            st.warning("Enter a clinical note first.")
            return

        with st.spinner("Processing..."):
            result = pipeline.process(text, redact=redact)
        st.success("Extraction complete")

        _render_metrics(result.entity_summary, len(result.phi))
        st.divider()

        if show_highlight and (result.entities or result.phi):
            st.markdown("**Annotated Text**")
            _render_legend()
            html = _annotated_text_html(result.normalized_text, result.entities, result.phi)
            st.markdown(
                f'<div style="background:#fafafa;padding:16px;border-radius:8px;'
                f'line-height:2.0;font-size:0.95em;color:#333;">{html}</div>',
                unsafe_allow_html=True,
            )
            st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Normalized Text**")
            st.text_area("normalized", value=result.normalized_text, height=180, label_visibility="collapsed")
        with col2:
            st.markdown("**Redacted Text**")
            st.text_area(
                "redacted",
                value=result.redacted_text or "(redaction disabled)",
                height=180,
                label_visibility="collapsed",
            )

        st.markdown("**Entities**")
        entities = [e.model_dump() for e in result.entities]
        if entities:
            st.dataframe(entities, use_container_width=True)
        else:
            st.info("No entities detected")

        st.markdown("**PHI Spans**")
        phi = [p.model_dump() for p in result.phi]
        if phi:
            st.dataframe(phi, use_container_width=True)
        else:
            st.info("No PHI spans detected")

        with st.expander("Sections"):
            for i, section in enumerate(result.sections):
                st.text(f"[{i+1}] {section[:200]}{'...' if len(section) > 200 else ''}")

        st.download_button(
            "Download JSON",
            data=result.model_dump_json(indent=2),
            file_name="clinical_nlp_result.json",
            mime="application/json",
        )


def render_batch(pipeline: ClinicalNLPPipeline) -> None:
    st.subheader("Batch Notes")
    st.caption("Separate notes using a line containing only `---`.")
    raw = st.text_area(
        "Batch input",
        value=(
            "Chief Complaint: Patient Jane Smith with heart failure takes furosemide 40 mg daily on 2024-03-01.\n"
            "Phone: 555-987-6543\n"
            "---\n"
            "History of Present Illness: Patient Michael Lee with chronic obstructive pulmonary disease "
            "uses albuterol 5 mg as needed. CT scan of the chest performed.\n"
            "SSN: 987-65-4321"
        ),
        height=200,
    )
    redact = st.checkbox("Redact PHI in batch", value=True, key="batch_redact")

    if st.button("Run Batch", key="run_batch", type="primary"):
        notes = split_notes(raw)
        if not notes:
            st.warning("Provide at least one note.")
            return

        with st.spinner(f"Processing {len(notes)} notes..."):
            results = [pipeline.process(note, redact=redact) for note in notes]

        outputs = [r.model_dump() for r in results]
        total_entities = sum(len(o["entities"]) for o in outputs)
        total_phi = sum(len(o["phi"]) for o in outputs)

        st.success(f"Processed {len(outputs)} notes")

        cols = st.columns(4)
        cols[0].metric("Notes", len(outputs))
        cols[1].metric("Total Entities", total_entities)
        cols[2].metric("Total PHI Spans", total_phi)
        type_counts: dict[str, int] = {}
        for o in outputs:
            for e in o["entities"]:
                type_counts[e["label"]] = type_counts.get(e["label"], 0) + 1
        cols[3].metric("Entity Types", len(type_counts))

        if type_counts:
            st.caption("Entity breakdown: " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())))

        for i, (result, note_text) in enumerate(zip(results, notes)):
            with st.expander(f"Note {i+1} ({len(result.entities)} entities, {len(result.phi)} PHI)"):
                _render_legend()
                html = _annotated_text_html(result.normalized_text, result.entities, result.phi)
                st.markdown(
                    f'<div style="background:#fafafa;padding:12px;border-radius:8px;'
                    f'line-height:2.0;font-size:0.9em;color:#333;">{html}</div>',
                    unsafe_allow_html=True,
                )
                st.json(outputs[i])

        st.download_button(
            "Download batch results JSON",
            data=json.dumps(outputs, indent=2),
            file_name="clinical_nlp_batch_results.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(page_title="Clinical NLP Pipeline", page_icon="🩺", layout="wide")
    st.title("Clinical NLP: PHI De-identification & Entity Extraction")
    st.caption(
        "End-to-end clinical note preprocessing, HIPAA-compliant de-identification (18 categories), "
        "and medical entity extraction (Diagnosis, Medication, Dosage, Procedure) with negation detection"
    )

    with st.spinner("Loading pipeline..."):
        pipeline = get_pipeline()

    tab1, tab2, tab3 = st.tabs(["Single Note", "Batch Notes", "About"])
    with tab1:
        render_single_note(pipeline)
    with tab2:
        render_batch(pipeline)
    with tab3:
        st.markdown("### Pipeline Architecture")
        st.markdown("""
**1. Preprocessing**
- Abbreviation expansion (600+ real clinical abbreviations)
- Section segmentation (25+ clinical section headers)
- Sentence splitting

**2. De-identification**
- All 18 HIPAA Safe Harbor categories
- Regex-based PHI detection with confidence scoring
- Greedy non-overlapping span selection

**3. Named Entity Recognition**
- Transformer-based NER (Bio_ClinicalBERT with BIO tagging)
- Heuristic dictionary NER (200+ medical terms, multi-word support)
- Ensemble merging with score-based selection

**4. Post-processing**
- Clinical negation detection (NegEx-style)
- Entity summary statistics
- Per-entity-type breakdown
        """)

        st.markdown("### Entity Types")
        for label, color in ENTITY_COLORS.items():
            st.markdown(
                f'<span style="background-color:{color};padding:3px 8px;'
                f'border-radius:4px;margin-right:8px;">{label}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("### Models")
        st.markdown("""
- **BiLSTM-CRF**: Bidirectional LSTM with real CRF layer (Viterbi decoding)
- **Transformer NER**: Fine-tunable Bio_ClinicalBERT for token classification
- **Heuristic NER**: Dictionary-based with multi-word entity support and dosage pattern matching
        """)


if __name__ == "__main__":
    main()
