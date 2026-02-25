from __future__ import annotations

import random
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# Realistic clinical vocabularies drawn from common EHR data
# ──────────────────────────────────────────────────────────────────────────────

DIAGNOSES = [
    # Cardiovascular
    "hypertension", "hyperlipidemia", "atrial fibrillation", "heart failure",
    "coronary artery disease", "acute myocardial infarction", "aortic stenosis",
    "mitral regurgitation", "peripheral arterial disease", "deep vein thrombosis",
    "pulmonary embolism", "cardiomyopathy", "endocarditis", "pericarditis",
    # Respiratory
    "pneumonia", "chronic obstructive pulmonary disease", "asthma", "bronchitis",
    "pulmonary fibrosis", "pleural effusion", "acute respiratory distress syndrome",
    "obstructive sleep apnea", "pulmonary hypertension", "lung cancer",
    # Endocrine / Metabolic
    "type 2 diabetes mellitus", "type 1 diabetes mellitus", "hypothyroidism",
    "hyperthyroidism", "diabetic ketoacidosis", "metabolic syndrome", "obesity",
    "gout", "osteoporosis", "adrenal insufficiency", "hypoglycemia",
    # GI
    "gastroesophageal reflux disease", "peptic ulcer disease", "cirrhosis",
    "hepatitis C", "pancreatitis", "cholecystitis", "diverticulitis",
    "inflammatory bowel disease", "celiac disease", "gastrointestinal hemorrhage",
    "small bowel obstruction", "appendicitis", "colorectal cancer",
    # Renal
    "chronic kidney disease", "acute kidney injury", "nephrolithiasis",
    "urinary tract infection", "glomerulonephritis", "polycystic kidney disease",
    # Neuro
    "stroke", "transient ischemic attack", "seizure disorder", "epilepsy",
    "migraine", "Parkinson disease", "Alzheimer disease", "multiple sclerosis",
    "peripheral neuropathy", "meningitis", "subarachnoid hemorrhage",
    "subdural hematoma", "traumatic brain injury",
    # Heme / Onc
    "iron deficiency anemia", "lymphoma", "leukemia", "multiple myeloma",
    "thrombocytopenia", "pancytopenia", "breast cancer", "prostate cancer",
    "melanoma", "colon cancer",
    # Infectious
    "sepsis", "cellulitis", "osteomyelitis", "HIV", "tuberculosis",
    "influenza", "COVID-19", "herpes zoster", "Clostridioides difficile infection",
    "MRSA bacteremia",
    # Psych
    "major depressive disorder", "generalized anxiety disorder", "bipolar disorder",
    "schizophrenia", "post-traumatic stress disorder", "substance use disorder",
    "alcohol use disorder",
    # MSK
    "osteoarthritis", "rheumatoid arthritis", "systemic lupus erythematosus",
    "lumbar radiculopathy", "rotator cuff tear", "fibromyalgia",
    "degenerative disc disease", "spinal stenosis",
]

MEDICATIONS = [
    # Cardiovascular
    "lisinopril", "amlodipine", "metoprolol", "atorvastatin", "rosuvastatin",
    "losartan", "valsartan", "hydrochlorothiazide", "furosemide", "spironolactone",
    "carvedilol", "diltiazem", "amiodarone", "digoxin", "hydralazine",
    "isosorbide mononitrate", "nitroglycerin", "apixaban", "rivaroxaban",
    "warfarin", "clopidogrel", "ticagrelor", "aspirin", "heparin", "enoxaparin",
    # Diabetes / Endocrine
    "metformin", "glipizide", "glyburide", "insulin glargine", "insulin lispro",
    "insulin aspart", "empagliflozin", "dapagliflozin", "sitagliptin",
    "semaglutide", "liraglutide", "pioglitazone", "levothyroxine",
    # GI
    "omeprazole", "pantoprazole", "esomeprazole", "famotidine", "ondansetron",
    "metoclopramide", "sucralfate", "lactulose", "polyethylene glycol",
    "docusate", "bisacodyl", "mesalamine",
    # Pain / Anti-inflammatory
    "acetaminophen", "ibuprofen", "naproxen", "ketorolac", "celecoxib",
    "morphine", "hydromorphone", "oxycodone", "fentanyl", "tramadol",
    "gabapentin", "pregabalin", "duloxetine", "lidocaine patch",
    # Antibiotics
    "amoxicillin", "amoxicillin-clavulanate", "azithromycin", "doxycycline",
    "ciprofloxacin", "levofloxacin", "ceftriaxone", "cefepime", "cephalexin",
    "piperacillin-tazobactam", "meropenem", "vancomycin", "metronidazole",
    "trimethoprim-sulfamethoxazole", "nitrofurantoin", "clindamycin", "linezolid",
    # Psych
    "sertraline", "fluoxetine", "escitalopram", "venlafaxine", "bupropion",
    "mirtazapine", "trazodone", "quetiapine", "olanzapine", "risperidone",
    "aripiprazole", "lithium", "valproic acid", "lamotrigine", "lorazepam",
    "alprazolam", "diazepam", "haloperidol",
    # Pulmonary
    "albuterol", "ipratropium", "fluticasone", "budesonide", "montelukast",
    "tiotropium", "prednisone", "methylprednisolone", "dexamethasone",
    # Other
    "potassium chloride", "calcium carbonate", "ferrous sulfate", "folic acid",
    "vitamin D", "magnesium oxide", "naloxone", "flumazenil", "epinephrine",
    "norepinephrine", "vasopressin", "propofol", "midazolam",
]

DOSAGES = [
    "2.5 mg", "5 mg", "10 mg", "12.5 mg", "20 mg", "25 mg", "40 mg", "50 mg",
    "75 mg", "80 mg", "100 mg", "125 mg", "150 mg", "200 mg", "250 mg",
    "300 mg", "325 mg", "400 mg", "500 mg", "600 mg", "750 mg", "800 mg",
    "1000 mg", "1 g", "2 g", "1.5 g",
    "0.25 mg", "0.5 mg", "1 mg", "2 mg", "4 mg",
    "5 mcg", "10 mcg", "25 mcg", "50 mcg", "75 mcg", "100 mcg", "125 mcg",
    "2 units", "4 units", "6 units", "8 units", "10 units", "12 units",
    "14 units", "16 units", "20 units", "30 units", "40 units",
    "2 mL", "5 mL", "10 mL", "15 mL", "30 mL",
    "2.5 mg/5 mL", "250 mg/5 mL",
    "0.5%", "1%", "2%",
]

PROCEDURES = [
    # Imaging
    "chest X-ray", "CT scan of the chest", "CT scan of the abdomen and pelvis",
    "CT angiography", "MRI of the brain", "MRI of the lumbar spine",
    "MRI of the knee", "transthoracic echocardiogram", "transesophageal echocardiogram",
    "abdominal ultrasound", "renal ultrasound", "Doppler ultrasound",
    "PET scan", "DEXA scan", "mammography", "bone scan",
    # Cardiac
    "cardiac catheterization", "coronary angiography", "percutaneous coronary intervention",
    "coronary artery bypass grafting", "pacemaker implantation",
    "stress test", "treadmill exercise test", "nuclear stress test",
    # GI
    "esophagogastroduodenoscopy", "colonoscopy", "flexible sigmoidoscopy",
    "ERCP", "paracentesis", "liver biopsy", "cholecystectomy", "appendectomy",
    # Pulmonary
    "bronchoscopy", "thoracentesis", "chest tube placement", "intubation",
    "tracheostomy", "pulmonary function testing",
    # Neuro
    "lumbar puncture", "EEG", "EMG and nerve conduction study",
    "carotid endarterectomy", "craniotomy",
    # Ortho
    "total hip arthroplasty", "total knee arthroplasty",
    "open reduction internal fixation", "arthroscopy", "spinal fusion",
    # Other
    "central line placement", "PICC line placement", "dialysis catheter placement",
    "bone marrow biopsy", "skin biopsy", "excisional biopsy",
    "incision and drainage", "wound debridement", "blood transfusion",
    "hemodialysis", "peritoneal dialysis",
]

NAMES = [
    "John Doe", "Jane Smith", "Michael Lee", "Anita Patel", "David Kim",
    "Maria Garcia", "James Wilson", "Sarah Johnson", "Robert Brown", "Lisa Chen",
    "William Davis", "Emily Rodriguez", "Christopher Martinez", "Amanda Taylor",
    "Daniel Anderson", "Jennifer Thomas", "Matthew Jackson", "Jessica White",
    "Anthony Harris", "Michelle Clark", "Andrew Lewis", "Stephanie Robinson",
    "Joshua Walker", "Nicole Hall", "Ryan Young", "Laura King", "Kevin Wright",
    "Rebecca Lopez", "Brian Hill", "Samantha Scott",
]

STREETS = [
    "101 Pine Street", "22 Oak Road", "850 Cedar Avenue", "77 Lake Boulevard",
    "445 Maple Drive", "1200 Washington Street", "33 Elm Court",
    "567 Broadway Avenue", "89 River Road", "234 Highland Avenue",
    "1001 Main Street", "55 Park Place", "700 Center Drive",
    "128 Willow Lane", "390 Spring Street",
]

DOCTORS = [
    "Dr. Sarah Mitchell", "Dr. James Patterson", "Dr. Priya Sharma",
    "Dr. Carlos Rivera", "Dr. Elizabeth Chen", "Dr. Michael Okafor",
    "Dr. Rachel Goldman", "Dr. Ahmed Khan", "Dr. Jennifer Park",
    "Dr. David Yamamoto", "Dr. Thomas Brennan", "Dr. Maria Santos",
]

ALLERGIES = [
    "penicillin", "sulfa drugs", "aspirin", "iodine contrast dye",
    "codeine", "morphine", "latex", "shellfish", "peanuts",
    "amoxicillin", "ciprofloxacin", "NSAIDs", "ACE inhibitors",
    "No Known Drug Allergies",
]

FREQUENCIES = [
    "once daily", "twice daily", "three times daily", "four times daily",
    "every 4 hours", "every 6 hours", "every 8 hours", "every 12 hours",
    "at bedtime", "as needed", "with meals", "before meals",
]

ROUTES = ["orally", "intravenously", "intramuscularly", "subcutaneously",
           "by inhalation", "topically", "sublingually"]

VITALS_TEMPLATES = [
    "Vitals: BP {bp}, HR {hr}, RR {rr}, Temp {temp} F, SpO2 {spo2}% on {o2}.",
    "Vital signs: Blood pressure {bp}, pulse {hr}, respirations {rr}, temperature {temp} degrees Fahrenheit, oxygen saturation {spo2}% on {o2}.",
    "VS: T {temp}, HR {hr}, BP {bp}, RR {rr}, O2 sat {spo2}% on {o2}.",
]

LAB_TEMPLATES = [
    "Labs notable for WBC {wbc}, Hgb {hgb}, Plt {plt}, Na {na}, K {k}, Cr {cr}, BUN {bun}, glucose {glucose}.",
    "Laboratory results: Sodium {na}, potassium {k}, creatinine {cr}, BUN {bun}, glucose {glucose}. CBC with WBC {wbc}, hemoglobin {hgb}, platelets {plt}.",
    "BMP: Na {na}, K {k}, Cl {cl}, CO2 {co2}, BUN {bun}, Cr {cr}, Glu {glucose}. CBC: WBC {wbc}, Hgb {hgb}, Hct {hct}, Plt {plt}.",
]

PHYSICAL_EXAM_TEMPLATES = [
    "Physical Exam: General: Alert, oriented, in {distress}. HEENT: {heent}. Cardiovascular: {cv}. Lungs: {lungs}. Abdomen: {abd}. Extremities: {ext}.",
    "On examination, the patient is {distress}. Head and neck exam is {heent}. Heart exam reveals {cv}. Lung auscultation demonstrates {lungs}. Abdominal exam is {abd}. Extremities show {ext}.",
]

DISTRESS_OPTIONS = ["no acute distress", "mild distress", "moderate distress"]
HEENT_OPTIONS = [
    "normocephalic, atraumatic, PERRL, EOMI, oropharynx clear",
    "mucous membranes moist, no lymphadenopathy, no JVD",
    "pupils equal and reactive, no scleral icterus, oropharynx clear",
]
CV_OPTIONS = [
    "regular rate and rhythm, no murmurs, rubs, or gallops",
    "irregular rhythm, systolic murmur at apex",
    "tachycardic, regular rhythm, S1 and S2 normal, no S3 or S4",
    "regular rate and rhythm with a 2/6 systolic murmur",
]
LUNGS_OPTIONS = [
    "clear to auscultation bilaterally",
    "bibasilar crackles, no wheezing",
    "decreased breath sounds at the left base",
    "scattered rhonchi bilaterally, no wheeze",
    "diminished breath sounds in the right lower lobe with dullness to percussion",
]
ABD_OPTIONS = [
    "soft, nontender, nondistended, bowel sounds present",
    "mild tenderness in the right lower quadrant, no rebound or guarding",
    "distended, tympanic, diffusely tender with guarding",
    "soft, mildly tender in the epigastric region, no hepatosplenomegaly",
]
EXT_OPTIONS = [
    "no edema, pulses intact bilaterally",
    "bilateral lower extremity pitting edema",
    "warm and well perfused, no cyanosis or clubbing",
    "1+ pitting edema to the ankles bilaterally",
]


def random_date(start_year: int = 2018, end_year: int = 2025) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    d = start + timedelta(days=random.randint(0, delta.days))
    return d.strftime("%m/%d/%Y")


def random_vitals() -> str:
    bp_sys = random.randint(90, 200)
    bp_dia = random.randint(50, 110)
    return random.choice(VITALS_TEMPLATES).format(
        bp=f"{bp_sys}/{bp_dia}",
        hr=random.randint(55, 130),
        rr=random.randint(12, 30),
        temp=round(random.uniform(97.0, 103.5), 1),
        spo2=random.randint(88, 100),
        o2=random.choice(["room air", "2L nasal cannula", "4L nasal cannula",
                          "high flow nasal cannula", "non-rebreather mask"]),
    )


def random_labs() -> str:
    return random.choice(LAB_TEMPLATES).format(
        wbc=round(random.uniform(2.0, 25.0), 1),
        hgb=round(random.uniform(6.0, 17.0), 1),
        hct=round(random.uniform(20.0, 52.0), 1),
        plt=random.randint(50, 500),
        na=random.randint(128, 150),
        k=round(random.uniform(2.8, 6.5), 1),
        cl=random.randint(95, 112),
        co2=random.randint(18, 32),
        cr=round(random.uniform(0.5, 8.0), 1),
        bun=random.randint(8, 80),
        glucose=random.randint(60, 400),
    )


def random_physical_exam() -> str:
    return random.choice(PHYSICAL_EXAM_TEMPLATES).format(
        distress=random.choice(DISTRESS_OPTIONS),
        heent=random.choice(HEENT_OPTIONS),
        cv=random.choice(CV_OPTIONS),
        lungs=random.choice(LUNGS_OPTIONS),
        abd=random.choice(ABD_OPTIONS),
        ext=random.choice(EXT_OPTIONS),
    )


def _find_all_occurrences(text: str, value: str) -> list[int]:
    """Find all start indices of value in text (case-insensitive)."""
    results = []
    lower_text = text.lower()
    lower_val = value.lower()
    start = 0
    while True:
        idx = lower_text.find(lower_val, start)
        if idx < 0:
            break
        results.append(idx)
        start = idx + 1
    return results


def make_note() -> dict:
    """Generate a realistic synthetic clinical note with entity annotations."""
    name = random.choice(NAMES)
    age = random.randint(18, 95)
    sex = random.choice(["male", "female"])

    num_dx = random.randint(1, 4)
    diagnoses = random.sample(DIAGNOSES, min(num_dx, len(DIAGNOSES)))
    primary_dx = diagnoses[0]

    num_meds = random.randint(1, 4)
    meds = random.sample(MEDICATIONS, min(num_meds, len(MEDICATIONS)))
    dosages = [random.choice(DOSAGES) for _ in meds]
    frequencies = [random.choice(FREQUENCIES) for _ in meds]
    routes = [random.choice(ROUTES) for _ in meds]

    num_proc = random.randint(0, 2)
    procedures = random.sample(PROCEDURES, min(num_proc, len(PROCEDURES))) if num_proc > 0 else []

    doctor = random.choice(DOCTORS)
    allergy = random.choice(ALLERGIES)
    date_admit = random_date()
    phone = f"({random.randint(200, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
    mrn = random.randint(10000000, 99999999)
    address = random.choice(STREETS)

    sections = []

    # Chief Complaint
    cc_templates = [
        f"Chief Complaint: {age}-year-old {sex} presenting with {primary_dx}.",
        f"Chief Complaint: Patient {name} is a {age} {sex} who presents with worsening {primary_dx}.",
        f"Chief Complaint: {primary_dx}.",
    ]
    sections.append(random.choice(cc_templates))

    # HPI
    hpi_parts = [f"History of Present Illness: Patient {name} is a {age}-year-old {sex} with a past medical history significant for "]
    hpi_parts.append(", ".join(diagnoses))
    hpi_parts.append(f" who presents to the emergency department on {date_admit} with complaints of worsening {primary_dx}. ")
    if len(diagnoses) > 1:
        hpi_parts.append(f"The patient also carries a diagnosis of {diagnoses[1]}. ")
    hpi_parts.append(f"The patient was evaluated by {doctor}. ")
    if meds:
        hpi_parts.append(f"Home medications include {meds[0]} {dosages[0]} {routes[0]} {frequencies[0]}. ")
    sections.append("".join(hpi_parts))

    # Allergies
    sections.append(f"Allergies: {allergy}.")

    # Medications
    med_lines = []
    for m, d, r, f in zip(meds, dosages, routes, frequencies):
        med_lines.append(f"{m} {d} {r} {f}")
    sections.append("Medications: " + "; ".join(med_lines) + ".")

    # Vitals
    sections.append(random_vitals())

    # Physical Exam
    sections.append(random_physical_exam())

    # Labs
    if random.random() > 0.3:
        sections.append(random_labs())

    # Assessment and Plan
    ap_parts = ["Assessment and Plan: "]
    for i, dx in enumerate(diagnoses, 1):
        ap_parts.append(f"{i}. {dx.capitalize()}")
        if i <= len(meds):
            ap_parts.append(f" - continue {meds[i - 1]} {dosages[i - 1]}")
        ap_parts.append(". ")
    if procedures:
        for proc in procedures:
            ap_parts.append(f"Plan for {proc}. ")
    ap_parts.append(f"Follow up with {doctor}. ")
    ap_parts.append(f"Contact: {phone}. MRN {mrn}. Address: {address}.")
    sections.append("".join(ap_parts))

    text = " ".join(sections)

    # Extract entity annotations with accurate offsets
    entities: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()

    def add_entities(value: str, label: str) -> None:
        for idx in _find_all_occurrences(text, value):
            span = (idx, idx + len(value))
            if span not in seen_spans:
                seen_spans.add(span)
                entities.append({
                    "start": idx,
                    "end": idx + len(value),
                    "text": text[idx: idx + len(value)],
                    "label": label,
                })

    for dx in diagnoses:
        add_entities(dx, "DIAGNOSIS")
    for m in meds:
        add_entities(m, "MEDICATION")
    for d in set(dosages):
        add_entities(d, "DOSAGE")
    for p in procedures:
        add_entities(p, "PROCEDURE")

    # Sort and deduplicate overlapping spans
    entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    deduped: list[dict] = []
    last_end = -1
    for ent in entities:
        if ent["start"] >= last_end:
            deduped.append(ent)
            last_end = ent["end"]

    return {"text": text, "entities": deduped}


def generate_dataset(num_notes: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    return [make_note() for _ in range(num_notes)]
