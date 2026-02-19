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


def split_notes(raw: str) -> list[str]:
    chunks = [chunk.strip() for chunk in raw.split("\n---\n")]
    return [chunk for chunk in chunks if chunk]


@st.cache_resource
def get_pipeline() -> ClinicalNLPPipeline:
    return ClinicalNLPPipeline()


def render_single_note(pipeline: ClinicalNLPPipeline) -> None:
    st.subheader("Single Note")
    text = st.text_area(
        "Clinical note",
        value=(
            "Chief Complaint: Patient John Doe presents with diabetes. "
            "History of Present Illness: Started metformin 500 mg daily on 01/03/2024."
        ),
        height=180,
    )

    redact = st.checkbox("Redact PHI", value=True)

    if st.button("Run Extraction", type="primary"):
        if not text.strip():
            st.warning("Enter a clinical note first.")
            return

        result = pipeline.process(text, redact=redact)
        st.success("Extraction complete")

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

        st.markdown("**Sections**")
        st.write(result.sections)

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
            "Patient Jane Smith with hypertension takes lisinopril 10 mg daily on 2024-03-01.\n"
            "---\n"
            "Patient Michael Lee with asthma uses albuterol 5 mg as needed."
        ),
        height=180,
    )
    redact = st.checkbox("Redact PHI in batch", value=True, key="batch_redact")

    if st.button("Run Batch", key="run_batch", type="primary"):
        notes = split_notes(raw)
        if not notes:
            st.warning("Provide at least one note.")
            return

        outputs = [pipeline.process(note, redact=redact).model_dump() for note in notes]

        st.success(f"Processed {len(outputs)} notes")
        total_entities = sum(len(o["entities"]) for o in outputs)
        total_phi = sum(len(o["phi"]) for o in outputs)
        st.write({"notes": len(outputs), "entities": total_entities, "phi_spans": total_phi})

        with st.expander("View first result"):
            st.json(outputs[0])

        st.download_button(
            "Download batch results JSON",
            data=json.dumps(outputs, indent=2),
            file_name="clinical_nlp_batch_results.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(page_title="Clinical NLP Demo", page_icon="🩺", layout="wide")
    st.title("Clinical NLP: PHI De-identification + NER")
    st.caption("Streamlit demo for clinical note preprocessing, de-identification, and entity extraction")

    with st.spinner("Loading pipeline..."):
        pipeline = get_pipeline()

    tab1, tab2 = st.tabs(["Single Note", "Batch Notes"])
    with tab1:
        render_single_note(pipeline)
    with tab2:
        render_batch(pipeline)


if __name__ == "__main__":
    main()
