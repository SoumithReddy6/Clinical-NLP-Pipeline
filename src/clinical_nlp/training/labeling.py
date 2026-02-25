from __future__ import annotations

import re

# BIO tagging scheme — industry standard for sequence labeling NER
LABELS = [
    "O",
    "B-DIAGNOSIS", "I-DIAGNOSIS",
    "B-MEDICATION", "I-MEDICATION",
    "B-DOSAGE", "I-DOSAGE",
    "B-PROCEDURE", "I-PROCEDURE",
]

DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|meq|%)\b", re.IGNORECASE
)
DOSAGE_CONTINUATION = re.compile(
    r"^(?:mg|mcg|g|ml|units?|meq|%|/\d+)$", re.IGNORECASE
)

DIAGNOSIS: set[str] = {
    "hypertension", "hyperlipidemia", "hypercholesterolemia", "dyslipidemia",
    "cardiomyopathy", "endocarditis", "pericarditis", "myocarditis",
    "tachycardia", "bradycardia", "arrhythmia",
    "pneumonia", "asthma", "bronchitis", "emphysema", "pleurisy",
    "diabetes", "hypothyroidism", "hyperthyroidism", "obesity",
    "osteoporosis", "gout", "hypoglycemia",
    "cirrhosis", "pancreatitis", "cholecystitis", "diverticulitis",
    "appendicitis", "hepatitis", "colitis", "gastritis",
    "nephrolithiasis", "glomerulonephritis", "pyelonephritis",
    "stroke", "epilepsy", "migraine", "meningitis", "encephalitis",
    "neuropathy", "dementia",
    "anemia", "lymphoma", "leukemia", "melanoma", "thrombocytopenia",
    "pancytopenia", "neutropenia",
    "sepsis", "cellulitis", "osteomyelitis", "tuberculosis",
    "influenza", "bacteremia",
    "schizophrenia", "depression", "anxiety", "insomnia",
    "osteoarthritis", "fibromyalgia", "stenosis",
}

DIAGNOSIS_MULTI: set[str] = {
    "atrial fibrillation", "heart failure", "coronary artery disease",
    "acute myocardial infarction", "aortic stenosis", "mitral regurgitation",
    "peripheral arterial disease", "deep vein thrombosis", "pulmonary embolism",
    "chronic obstructive pulmonary disease", "pulmonary fibrosis",
    "pleural effusion", "acute respiratory distress syndrome",
    "obstructive sleep apnea", "pulmonary hypertension", "lung cancer",
    "type 2 diabetes mellitus", "type 1 diabetes mellitus",
    "diabetic ketoacidosis", "metabolic syndrome", "adrenal insufficiency",
    "gastroesophageal reflux disease", "peptic ulcer disease",
    "hepatitis C", "inflammatory bowel disease", "celiac disease",
    "gastrointestinal hemorrhage", "small bowel obstruction",
    "colorectal cancer", "chronic kidney disease", "acute kidney injury",
    "urinary tract infection", "polycystic kidney disease",
    "transient ischemic attack", "seizure disorder", "Parkinson disease",
    "Alzheimer disease", "multiple sclerosis", "peripheral neuropathy",
    "subarachnoid hemorrhage", "subdural hematoma", "traumatic brain injury",
    "iron deficiency anemia", "multiple myeloma", "breast cancer",
    "prostate cancer", "colon cancer", "herpes zoster",
    "major depressive disorder", "generalized anxiety disorder",
    "bipolar disorder", "post-traumatic stress disorder",
    "substance use disorder", "alcohol use disorder",
    "rheumatoid arthritis", "systemic lupus erythematosus",
    "lumbar radiculopathy", "rotator cuff tear", "degenerative disc disease",
    "spinal stenosis",
}

MEDICATION: set[str] = {
    "lisinopril", "amlodipine", "metoprolol", "atorvastatin", "rosuvastatin",
    "losartan", "valsartan", "hydrochlorothiazide", "furosemide", "spironolactone",
    "carvedilol", "diltiazem", "amiodarone", "digoxin", "hydralazine",
    "nitroglycerin", "apixaban", "rivaroxaban", "warfarin", "clopidogrel",
    "ticagrelor", "aspirin", "heparin", "enoxaparin",
    "metformin", "glipizide", "glyburide", "empagliflozin", "dapagliflozin",
    "sitagliptin", "semaglutide", "liraglutide", "pioglitazone", "levothyroxine",
    "omeprazole", "pantoprazole", "esomeprazole", "famotidine", "ondansetron",
    "metoclopramide", "sucralfate", "lactulose", "docusate", "bisacodyl",
    "mesalamine",
    "acetaminophen", "ibuprofen", "naproxen", "ketorolac", "celecoxib",
    "morphine", "hydromorphone", "oxycodone", "fentanyl", "tramadol",
    "gabapentin", "pregabalin", "duloxetine",
    "amoxicillin", "azithromycin", "doxycycline", "ciprofloxacin",
    "levofloxacin", "ceftriaxone", "cefepime", "cephalexin", "meropenem",
    "vancomycin", "metronidazole", "nitrofurantoin", "clindamycin", "linezolid",
    "sertraline", "fluoxetine", "escitalopram", "venlafaxine", "bupropion",
    "mirtazapine", "trazodone", "quetiapine", "olanzapine", "risperidone",
    "aripiprazole", "lithium", "lamotrigine", "lorazepam", "alprazolam",
    "diazepam", "haloperidol",
    "albuterol", "ipratropium", "fluticasone", "budesonide", "montelukast",
    "tiotropium", "prednisone", "methylprednisolone", "dexamethasone",
    "naloxone", "flumazenil", "epinephrine", "norepinephrine", "vasopressin",
    "propofol", "midazolam",
}

MEDICATION_MULTI: set[str] = {
    "insulin glargine", "insulin lispro", "insulin aspart",
    "isosorbide mononitrate", "amoxicillin-clavulanate",
    "piperacillin-tazobactam", "trimethoprim-sulfamethoxazole",
    "valproic acid", "lidocaine patch", "polyethylene glycol",
    "potassium chloride", "calcium carbonate", "ferrous sulfate",
    "folic acid", "vitamin D", "magnesium oxide",
}

PROCEDURE: set[str] = {
    "colonoscopy", "bronchoscopy", "appendectomy", "cholecystectomy",
    "esophagogastroduodenoscopy", "thoracentesis", "paracentesis",
    "tracheostomy", "intubation", "craniotomy", "arthroscopy",
    "mammography", "hemodialysis", "ERCP", "EEG",
}

PROCEDURE_MULTI: set[str] = {
    "chest X-ray", "CT scan", "CT angiography", "MRI of the brain",
    "MRI of the lumbar spine", "MRI of the knee",
    "transthoracic echocardiogram", "transesophageal echocardiogram",
    "abdominal ultrasound", "renal ultrasound", "Doppler ultrasound",
    "PET scan", "DEXA scan", "bone scan",
    "cardiac catheterization", "coronary angiography",
    "percutaneous coronary intervention", "coronary artery bypass grafting",
    "pacemaker implantation", "stress test", "treadmill exercise test",
    "nuclear stress test", "flexible sigmoidoscopy", "liver biopsy",
    "chest tube placement", "pulmonary function testing",
    "lumbar puncture", "EMG and nerve conduction study",
    "carotid endarterectomy", "total hip arthroplasty",
    "total knee arthroplasty", "open reduction internal fixation",
    "spinal fusion", "central line placement", "PICC line placement",
    "dialysis catheter placement", "bone marrow biopsy", "skin biopsy",
    "excisional biopsy", "incision and drainage", "wound debridement",
    "blood transfusion", "peritoneal dialysis",
}


def token_level_labels(tokens: list[str]) -> list[str]:
    """Assign BIO labels to a token sequence using dictionary + pattern matching."""
    n = len(tokens)
    labels = ["O"] * n
    used = [False] * n

    # Multi-word entity matching (longest match first)
    for multi_set, b_tag, i_tag in [
        (DIAGNOSIS_MULTI, "B-DIAGNOSIS", "I-DIAGNOSIS"),
        (MEDICATION_MULTI, "B-MEDICATION", "I-MEDICATION"),
        (PROCEDURE_MULTI, "B-PROCEDURE", "I-PROCEDURE"),
    ]:
        for phrase in sorted(multi_set, key=len, reverse=True):
            phrase_tokens = phrase.lower().split()
            plen = len(phrase_tokens)
            for i in range(n - plen + 1):
                if any(used[i + j] for j in range(plen)):
                    continue
                match = all(
                    tokens[i + j].lower().rstrip(".,;:!?") == phrase_tokens[j]
                    for j in range(plen)
                )
                if match:
                    labels[i] = b_tag
                    for j in range(1, plen):
                        labels[i + j] = i_tag
                    for j in range(plen):
                        used[i + j] = True

    # Single-word entity matching
    for i in range(n):
        if used[i]:
            continue
        t = tokens[i].lower().rstrip(".,;:!?")
        if t in DIAGNOSIS:
            labels[i] = "B-DIAGNOSIS"
            used[i] = True
        elif t in MEDICATION:
            labels[i] = "B-MEDICATION"
            used[i] = True
        elif t in PROCEDURE:
            labels[i] = "B-PROCEDURE"
            used[i] = True
        elif DOSAGE_PATTERN.match(tokens[i]):
            labels[i] = "B-DOSAGE"
            used[i] = True

    # Dosage continuation tokens (e.g., "500" then "mg")
    for i in range(1, n):
        if not used[i] and labels[i - 1] in ("B-DOSAGE", "I-DOSAGE"):
            if DOSAGE_CONTINUATION.match(tokens[i].rstrip(".,;:!?")):
                labels[i] = "I-DOSAGE"
                used[i] = True

    return labels
