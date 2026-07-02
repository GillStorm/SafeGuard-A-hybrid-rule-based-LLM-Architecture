import pandas as pd
import os
import xml.etree.ElementTree as ET
import re


# ---------- CLEAN FUNCTION ----------
def clean_query(q):
    q = str(q).lower()
    q = re.sub(r"what is|what are|tell me about|define", "", q)
    q = re.sub(r"[^a-zA-Z\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


# ---------- MEDQUAD ----------
def load_medquad(folder):
    data = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".xml"):
                path = os.path.join(root, file)
                try:
                    tree = ET.parse(path)
                    root_xml = tree.getroot()

                    for elem in root_xml.iter():
                        if elem.tag == "Question" and elem.text:
                            data.append(elem.text.strip())
                except:
                    pass

    return data


# ---------- TRUTHFUL QA ----------
def load_truthfulqa(csv_path):
    df = pd.read_csv(csv_path)

    # handle column name variations
    col = "Question" if "Question" in df.columns else df.columns[0]
    return df[col].dropna().tolist()


# ---------- KAGGLE SEVERITY ----------
def load_symptom_severity(csv_path):
    df = pd.read_csv(csv_path)

    severity_map = {}

    for _, row in df.iterrows():
        for col in df.columns:
            symptom = str(row[col]).lower().strip()
            if symptom != "nan":
                severity_map[symptom] = 0.7  # default weight

    print("Loaded symptoms:", len(severity_map))
    return severity_map


# ---------- LABEL ----------
def assign_label(q, severity_map):
    q_lower = q.lower()

    # Critical rules
    if any(x in q_lower for x in [
        "chest pain", "not breathing", "shortness of breath",
        "unconscious", "severe bleeding"
    ]):
        return "Critical", 0.9

    # Kaggle symptom mapping
    for symptom in severity_map:
        if symptom in q_lower:
            return "Ambiguous", severity_map[symptom]

    return "Non-Critical", 0.2


# ---------- BUILD DATASET ----------
def build_dataset():
    all_queries = []

    # 🔴 UPDATE PATHS HERE
    medquad_path = r"MedQuAD-master"
    truthfulqa_path = r"TruthfulQA-main\TruthfulQA-main\TruthfulQA.csv"
    kaggle_path = r"archive (2)\Symptom-severity.csv"

    # Load data
    medquad = load_medquad(medquad_path)
    truthful = load_truthfulqa(truthfulqa_path)
    severity_map = load_symptom_severity(kaggle_path)

    print("MedQuAD:", len(medquad))
    print("TruthfulQA:", len(truthful))

    all_queries.extend(medquad)
    all_queries.extend(truthful)

    df = pd.DataFrame(all_queries, columns=["query"])

    # Clean
    df["query"] = df["query"].apply(clean_query)

    # Remove empty + duplicates
    df = df[df["query"] != ""]
    df = df.drop_duplicates()

    # Label
    labels = df["query"].apply(lambda q: assign_label(q, severity_map))

    df["label"] = labels.apply(lambda x: x[0])
    df["severity"] = labels.apply(lambda x: x[1])
    df["domain"] = "Medical"

    # Save
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/final_dataset.csv", index=False)

    print("Dataset created:", len(df))
    print(df.head())


if __name__ == "__main__":
    build_dataset()