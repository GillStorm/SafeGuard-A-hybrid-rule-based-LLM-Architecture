# core/session_state.py
"""
Conversational Symptom Accumulation (Level 2)
Tracks symptoms across conversation turns and triggers emergency
overrides when dangerous symptom combinations are detected.

Makes the system STATEFUL — 99% of university projects miss this.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.llm import client, MODEL_NAME, GROQ_READY

logger = logging.getLogger(__name__)


# ==========================================
# SYMPTOM CLUSTERS
# ==========================================
# Dangerous symptom combinations that individually seem Ambiguous
# but together indicate a medical emergency.

SYMPTOM_CLUSTERS = {
    "cardiac": {
        "symptoms": {
            "arm pain/weakness/numbness",
            "jaw pain/ache",
            "chest pressure/discomfort",
            "sweating/clammy",
            "nausea",
            "shortness of breath",
            "shoulder/back pain",
            "dizzy/lightheaded",
        },
        "threshold": 2,  # 2 co-occurring symptoms = cardiac alert
        "response": (
            "🚨 ESCALATION ALERT: Based on your symptoms across this conversation "
            "({symptoms}), there is a pattern consistent with a potential cardiac event. "
            "Please call 911 or go to the emergency room IMMEDIATELY. "
            "Do NOT wait for symptoms to worsen."
        ),
    },
    "stroke": {
        "symptoms": {
            "face drooping", "facial droop", "face numb",
            "arm weakness", "arm numbness", "leg weakness", "leg numbness",
            "speech difficulty", "slurred speech", "cant speak",
            "vision loss", "sudden blindness", "blurred vision sudden",
            "sudden confusion", "disoriented", "sudden headache",
            "one side weak", "one side numb",
        },
        "threshold": 2,
        "response": (
            "🚨 ESCALATION ALERT: Your symptoms across this conversation "
            "({symptoms}) match the pattern of a possible stroke. "
            "Remember FAST: Face, Arms, Speech, Time. "
            "Call 911 IMMEDIATELY. Note the time symptoms started."
        ),
    },
    "sepsis": {
        "symptoms": {
            "high fever", "fever", "temperature high",
            "rapid heartbeat", "heart racing", "fast pulse",
            "confusion", "confused", "disoriented",
            "difficulty breathing", "shortness of breath",
            "extreme fatigue", "cant stay awake",
            "chills", "shivering",
        },
        "threshold": 3,  # Sepsis needs more co-occurring symptoms
        "response": (
            "🚨 ESCALATION ALERT: The combination of symptoms you have described "
            "({symptoms}) could indicate sepsis, a life-threatening condition. "
            "Please seek emergency medical care IMMEDIATELY."
        ),
    },
}

# Keyword patterns to extract symptoms from natural language
SYMPTOM_EXTRACTION_PATTERNS = {
    # Cardiac
    "arm pain/weakness/numbness": [r"(arm|arms).*(pain|hurt|ache|sore|weird|strange|tingle|numb|weak|heavy|cant move)"],
    "jaw pain/ache": [r"(jaw|jaws).*(pain|hurt|ache|sore|throb)"],
    "chest pressure/discomfort": [r"(chest|thorax).*(tight|pressure|squeez|constrict|discomfort|uncomfortable|weird|strange|heavy|weight)"],
    "sweating/clammy": [r"(sweat|perspir|clammy|cold sweat|drenched)"],
    "nausea": [r"(nausea|nauseous|queasy|feel sick|going to vomit|want to throw up)"],
    "shortness of breath": [r"(short.?ness.*breath|breathless|cant.*breath|hard.*breath|difficult.*breath)"],
    "shoulder/back pain": [r"(shoulder|back|between.*shoulder).*(pain|hurt|ache)"],
    "dizzy/lightheaded": [r"(lightheaded|light.?headed|woozy|dizz|vertigo)"],

    # Stroke
    "face drooping": [r"(face|facial).*(droop|sag|lopsided|one side)"],
    "face numb": [r"(face|facial).*(numb|tingle|cant feel)"],
    "speech difficulty": [r"(speech|speak|talk).*(slur|garble|difficult|hard|trouble|cant)"],
    "slurred speech": [r"(slur|garble).*(speech|word|talk)"],
    "vision loss": [r"(vision|sight|see).*(loss|lost|cant|blur|sudden)"],
    "sudden confusion": [r"(sudden|suddenly).*(confus|disoriented|lost)"],
    "sudden headache": [r"(sudden|suddenly|worst).*(headache|head pain)"],
    "one side weak": [r"(one side|left side|right side).*(weak|numb|cant move|heavy)"],
    "leg weakness": [r"(leg|legs).*(weak|heavy|cant move|gave out)"],
    "leg numbness": [r"(leg|legs).*(numb|tingl|cant feel)"],

    # Sepsis
    "high fever": [r"(high|very|extreme).*(fever|temperature)", r"(fever|temperature).*(high|very|\d{3})"],
    "fever": [r"(fever|febrile|temperature)"],
    "rapid heartbeat": [r"(heart|pulse).*(rac|fast|rapid|pound|flutter)"],
    "confusion": [r"(confus|disoriented|cant think straight)"],
    "extreme fatigue": [r"(extreme|severe|total).*(fatigue|exhausti|tired|weak)"],
    "chills": [r"(chill|shiver|shak|rigor)"],
}


@dataclass
class SessionState:
    """Tracks conversation state across turns."""
    detected_symptoms: List[str] = field(default_factory=list)
    symptom_turns: Dict[str, int] = field(default_factory=dict)  # symptom -> turn number
    turn_count: int = 0
    risk_history: List[float] = field(default_factory=list)
    escalation_triggered: bool = False
    triggered_cluster: str = ""


@dataclass
class SessionUpdate:
    """Result of updating the session with a new query."""
    new_symptoms: List[str]
    total_symptoms: List[str]
    escalation_triggered: bool
    escalation_response: str
    triggered_cluster: str
    cluster_matches: Dict[str, int]  # cluster_name -> number of matching symptoms


def _extract_symptoms_regex(query: str) -> List[str]:
    """
    Extract symptom mentions from a query using regex patterns.
    Returns a list of canonical symptom names found.
    """
    query_lower = query.lower()
    found = []

    for symptom_name, patterns in SYMPTOM_EXTRACTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                if symptom_name not in found:
                    found.append(symptom_name)
                break  # One match per symptom is enough

    return found

def extract_symptoms(query: str) -> List[str]:
    """
    Extract symptom mentions from a query using Groq LLM JSON mode.
    Falls back to regex if LLM is unavailable or fails.
    """
    if not GROQ_READY or not client:
        return _extract_symptoms_regex(query)

    valid_symptoms = set()
    for cluster in SYMPTOM_CLUSTERS.values():
        valid_symptoms.update(cluster["symptoms"])
    valid_symptoms = sorted(list(valid_symptoms))
    
    prompt = f"""You are a medical symptom extractor.
Your task is to identify which of the following canonical symptoms are present in the user's text.
Canonical symptoms list:
{valid_symptoms}

User text: "{query}"

Output ONLY a JSON object with a single key "symptoms" containing a list of strings of the exact canonical symptoms found.
Example: {{"symptoms": ["chest pressure/discomfort", "nausea"]}}
If none are found, output {{"symptoms": []}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        found = data.get("symptoms", [])
        return [s for s in found if s in valid_symptoms]
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return _extract_symptoms_regex(query)


def update_session(
    query: str,
    classification: dict,
    session: SessionState,
) -> SessionUpdate:
    """
    Update session state with a new query.

    1. Extract symptoms from the query
    2. Add new symptoms to the running list
    3. Check if any symptom cluster threshold is crossed
    4. Return whether an emergency escalation should fire

    Args:
        query: The current user query.
        classification: Dict with 'label', 'severity' etc.
        session: The current session state (mutated in place).

    Returns:
        SessionUpdate with escalation info.
    """
    session.turn_count += 1

    # Extract symptoms from this turn
    new_symptoms = extract_symptoms(query)

    # Add new symptoms (avoid duplicates)
    for symptom in new_symptoms:
        if symptom not in session.detected_symptoms:
            session.detected_symptoms.append(symptom)
            session.symptom_turns[symptom] = session.turn_count

    # Track risk
    severity = classification.get("severity", 0.0)
    session.risk_history.append(severity)

    # Check symptom clusters
    cluster_matches = {}
    escalation_triggered = False
    escalation_response = ""
    triggered_cluster = ""

    if not session.escalation_triggered:  # Don't re-trigger
        for cluster_name, cluster_data in SYMPTOM_CLUSTERS.items():
            # Count how many cluster symptoms the user has mentioned
            matched = [
                s for s in session.detected_symptoms
                if s in cluster_data["symptoms"]
            ]
            cluster_matches[cluster_name] = len(matched)

            if len(matched) >= cluster_data["threshold"]:
                escalation_triggered = True
                triggered_cluster = cluster_name
                symptom_list = ", ".join(matched)
                escalation_response = cluster_data["response"].format(
                    symptoms=symptom_list
                )
                session.escalation_triggered = True
                session.triggered_cluster = cluster_name
                logger.warning(
                    f"ESCALATION: {cluster_name} cluster triggered with "
                    f"symptoms: {matched}"
                )
                break

    return SessionUpdate(
        new_symptoms=new_symptoms,
        total_symptoms=list(session.detected_symptoms),
        escalation_triggered=escalation_triggered,
        escalation_response=escalation_response,
        triggered_cluster=triggered_cluster,
        cluster_matches=cluster_matches,
    )


def reset_session(session: SessionState):
    """Reset session state (e.g., on 'clear' command)."""
    session.detected_symptoms.clear()
    session.symptom_turns.clear()
    session.turn_count = 0
    session.risk_history.clear()
    session.escalation_triggered = False
    session.triggered_cluster = ""


if __name__ == "__main__":
    print("Testing Conversational Symptom Accumulation\n")
    print("=" * 60)

    session = SessionState()

    # Simulate multi-turn conversation
    turns = [
        ("My left arm feels weird and tingly", {"label": "Ambiguous", "severity": 0.5}),
        ("Now my jaw is starting to hurt too", {"label": "Ambiguous", "severity": 0.5}),
    ]

    for query, classification in turns:
        print(f"\nTurn {session.turn_count + 1}: \"{query}\"")
        result = update_session(query, classification, session)
        print(f"  New symptoms: {result.new_symptoms}")
        print(f"  All symptoms: {result.total_symptoms}")
        print(f"  Cluster matches: {result.cluster_matches}")

        if result.escalation_triggered:
            print(f"\n  🚨 ESCALATION TRIGGERED: {result.triggered_cluster}")
            print(f"  {result.escalation_response}")
