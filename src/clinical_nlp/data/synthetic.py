from __future__ import annotations

import random
from datetime import datetime, timedelta

DIAGNOSES = ["diabetes", "hypertension", "asthma", "pneumonia", "sepsis", "copd"]
MEDICATIONS = ["metformin", "lisinopril", "atorvastatin", "albuterol", "insulin", "aspirin"]
DOSAGES = ["5 mg", "10 mg", "20 mg", "500 mg", "1 g", "2 units"]
PROCEDURES = ["MRI", "CT", "X-ray", "colonoscopy", "appendectomy", "biopsy"]
NAMES = ["John Doe", "Jane Smith", "Michael Lee", "Anita Patel", "David Kim"]
STREETS = ["101 Pine St", "22 Oak Road", "850 Cedar Ave", "77 Lake Blvd"]


def random_date() -> str:
    start = datetime(2018, 1, 1)
    end = datetime(2025, 12, 31)
    delta = end - start
    d = start + timedelta(days=random.randint(0, delta.days))
    return d.strftime("%m/%d/%Y")


def make_note() -> dict:
    name = random.choice(NAMES)
    diagnosis = random.choice(DIAGNOSES)
    med = random.choice(MEDICATIONS)
    dosage = random.choice(DOSAGES)
    procedure = random.choice(PROCEDURES)
    date = random_date()
    phone = f"555-{random.randint(100,999)}-{random.randint(1000,9999)}"
    mrn = random.randint(10000000, 99999999)
    address = random.choice(STREETS)

    text = (
        f"Chief Complaint: Patient {name} presents with {diagnosis}. "
        f"History of Present Illness: Started {med} {dosage} daily on {date}. "
        f"Medications: {med} {dosage}. "
        f"Assessment: Persistent {diagnosis}. "
        f"Plan: Schedule {procedure}. Contact {phone}. MRN {mrn}. Address {address}."
    )

    entities = []
    for label, value in [
        ("DIAGNOSIS", diagnosis),
        ("MEDICATION", med),
        ("DOSAGE", dosage),
        ("PROCEDURE", procedure),
    ]:
        idx = text.lower().find(value.lower())
        if idx >= 0:
            entities.append({"start": idx, "end": idx + len(value), "text": text[idx : idx + len(value)], "label": label})

    return {"text": text, "entities": entities}


def generate_dataset(num_notes: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    return [make_note() for _ in range(num_notes)]
