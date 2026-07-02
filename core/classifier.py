import re


# Medical keyword lists for classification
CRITICAL_KEYWORDS = [
    "chest pain", "shortness of breath", "not breathing",
    "heart attack", "stroke", "seizure", "unconscious",
    "severe bleeding", "difficulty breathing", "cardiac arrest"
]

AMBIGUOUS_KEYWORDS = [
    "dizziness", "fever", "headache", "nausea",
    "fatigue", "tired", "weakness", "vomiting",
    "stomach pain", "back pain", "sore throat"
]

MEDICAL_KEYWORDS = [
    "pain", "breathing", "cough", "infection", "disease",
    "symptom", "treatment", "medication", "doctor", "hospital",
    "blood", "heart", "lung", "kidney", "liver", "cancer",
    "diabetes", "allergy", "surgery", "diagnosis", "therapy",
    "fever", "nausea", "headache", "dizziness", "chest",
    "injury", "fracture", "swelling", "rash", "inflammation",
    "tired", "fatigue", "weakness", "vomiting", "sore throat",
    "stomach", "seizure", "unconscious", "bleeding", "stroke"
]


def clean_input(query):
    """Normalize and clean the input query."""
    q = query.lower().strip()
    q = re.sub(r"[^a-zA-Z\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def is_medical(query):
    """Check if the query is related to the medical domain."""
    q = clean_input(query)
    return any(keyword in q for keyword in MEDICAL_KEYWORDS)


def get_severity(query):
    """Calculate severity score based on keyword matches."""
    q = clean_input(query)

    if any(kw in q for kw in CRITICAL_KEYWORDS):
        return 0.9

    if any(kw in q for kw in AMBIGUOUS_KEYWORDS):
        return 0.5

    return 0.2


def classify(query):
    """
    Classify a query into:
      - label: Critical / Ambiguous / Non-Critical / Off-Topic
      - severity: 0.0 to 0.9
      - domain: Medical or General

    Returns a dictionary with classification results.
    """
    q = clean_input(query)

    # Off-Topic check (not medical at all)
    if not is_medical(q):
        return {
            "query": query,
            "clean_query": q,
            "label": "Off-Topic",
            "severity": 0.0,
            "domain": "General",
            "action": "REJECTED -- not a medical query."
        }

    # Critical
    if any(kw in q for kw in CRITICAL_KEYWORDS):
        return {
            "query": query,
            "clean_query": q,
            "label": "Critical",
            "severity": 0.9,
            "domain": "Medical",
            "action": "URGENT -- Seek immediate medical attention!"
        }

    # Ambiguous
    if any(kw in q for kw in AMBIGUOUS_KEYWORDS):
        return {
            "query": query,
            "clean_query": q,
            "label": "Ambiguous",
            "severity": 0.5,
            "domain": "Medical",
            "action": "CAUTION -- Monitor symptoms, consult a doctor if they persist."
        }

    # Non-Critical (medical but low severity)
    return {
        "query": query,
        "clean_query": q,
        "label": "Non-Critical",
        "severity": 0.2,
        "domain": "Medical",
        "action": "LOW RISK -- general medical information."
    }

if __name__ == "__main__":
    print("Testing classifier...")
    test_queries = [
        "I have severe chest pain",
        "I feel very tired and fatigued",
        "what is the meaning of life"
    ]
    for q in test_queries:
        res = classify(q)
        print(f"\nQuery: {q}")
        print(f"Label: {res['label']} (Severity: {res['severity']})")
        print(f"Action: {res['action']}")
