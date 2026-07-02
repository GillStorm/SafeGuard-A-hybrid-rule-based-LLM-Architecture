import os
import xml.etree.ElementTree as ET
import pandas as pd
import re

# 🔴 UPDATE THIS PATH (IMPORTANT)
folder = r"C:\Users\ashis\OneDrive\Documents\Important stuff of AG\Logical_LLm\MedQuAD-master"


# 🧠 Extract Questions from XML (FIXED)
def extract_medquad(folder):
    data = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".xml"):
                path = os.path.join(root, file)

                try:
                    tree = ET.parse(path)
                    root_xml = tree.getroot()

                    # 🔥 FIX: Use iter() instead of QAPair
                    for elem in root_xml.iter():
                        if elem.tag == "Question":
                            if elem.text:
                                data.append(elem.text.strip())

                except Exception as e:
                    print(f"Error reading {file}: {e}")

    df = pd.DataFrame(data, columns=["query"])
    return df


# 🧹 Clean Query
def clean_query(q):
    q = q.lower()
    q = re.sub(r"what is|what are|what causes|tell me about|define", "", q)
    q = re.sub(r"\(.*?\)", "", q)
    q = re.sub(r"[^a-zA-Z\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


# 🔍 Filter useful queries
def is_useful(q):
    keywords = [
        "pain", "breathing", "dizziness",
        "fever", "headache", "cough",
        "nausea", "chest"
    ]
    return any(k in q for k in keywords)


# 🏷️ Assign label + severity
def assign_label(q):
    if any(x in q for x in ["chest pain", "shortness of breath", "not breathing"]):
        return "Critical", 0.9

    elif any(x in q for x in ["dizziness", "fever", "headache"]):
        return "Ambiguous", 0.5

    else:
        return "Non-Critical", 0.2


# 🚀 Build Dataset
def build_dataset(folder):
    df = extract_medquad(folder)

    print("Total extracted:", len(df))

    # 🔴 SAFETY CHECK
    if df.empty:
        print("ERROR: No data extracted. Check folder path or XML parsing.")
        return

    # Clean
    df["query"] = df["query"].apply(clean_query)

    # Filter
    df = df[df["query"].apply(is_useful)]

    print("After filtering:", len(df))

    # Labeling
    labels = df["query"].apply(assign_label)
    df["label"] = labels.apply(lambda x: x[0])
    df["severity"] = labels.apply(lambda x: x[1])
    df["domain"] = "Medical"

    # Create folder if not exists
    os.makedirs("data", exist_ok=True)

    # Save
    df.to_csv("data/final_dataset.csv", index=False)

    print("✅ Dataset saved successfully!")
    print(df.head())


# ▶️ Run
if __name__ == "__main__":
    build_dataset(folder)