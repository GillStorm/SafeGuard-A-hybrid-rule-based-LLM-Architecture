# core/rules.py
"""
Medical Safety Rules Engine
Overrides LLM responses for emergency patterns
"""

from typing import Optional, List, Dict, Tuple
import re
from dataclasses import dataclass
from enum import Enum


class EmergencyLevel(Enum):
    IMMEDIATE = "immediate"  # Call 911 now
    URGENT = "urgent"        # Go to ER within hours
    CONSULT = "consult"      # See doctor soon
    NONE = "none"            # No emergency


@dataclass
class RuleMatch:
    """Result of rule matching"""
    matched: bool
    level: EmergencyLevel
    response: str
    matched_patterns: List[str]
    category: str


# Emergency patterns with severity levels
EMERGENCY_RULES: Dict[str, Dict] = {
    # IMMEDIATE - Call 911
    "chest_pain_emergency": {
        "patterns": [
            r"chest pain.*(can't|cannot|unable to) breathe",
            r"(severe|crushing|extreme) chest pain",
            r"chest pain.*(radiating|spreading).*(arm|jaw|back)",
            r"chest pain.*(sweating|nausea|dizzy)",
            r"heart attack",
            r"cardiac arrest",
        ],
        "level": EmergencyLevel.IMMEDIATE,
        "response": "🚨 EMERGENCY: Call 911 immediately! This could be a heart attack. " \
                   "Do NOT drive yourself. Sit or lie down. Chew an aspirin if available " \
                   "and not allergic. Unlock your door for paramedics.",
        "category": "cardiac"
    },
    "breathing_emergency": {
        "patterns": [
            r"(can't|cannot|unable to) breathe",
            r"(severe|extreme) difficulty breathing",
            r"choking",
            r"(throat|airway).*(closing|swelling|blocked)",
            r"(not breathing|stopped breathing)",
        ],
        "level": EmergencyLevel.IMMEDIATE,
        "response": "🚨 EMERGENCY: Call 911 immediately! Check if person is conscious. " \
                   "If choking and conscious, perform Heimlich maneuver. " \
                   "If not breathing, begin CPR if trained. Stay on line with 911.",
        "category": "respiratory"
    },
    "unconscious_emergency": {
        "patterns": [
            r"(unconscious|unresponsive|passed out|collapsed)",
            r"(can't|cannot|unable to).*(wake|wake up)",
            r"not (responding|responsive)",
            r"(found|found him|found her).*(unconscious|unresponsive)",
        ],
        "level": EmergencyLevel.IMMEDIATE,
        "response": "🚨 EMERGENCY: Call 911 immediately! Check for breathing and pulse. " \
                   "If not breathing, start CPR. Place in recovery position if breathing. " \
                   "Do NOT leave the person alone. Stay on line with dispatch.",
        "category": "neurological"
    },
    "stroke_emergency": {
        "patterns": [
            r"(face|arm|leg).*(droop|drooping|weakness|numb)",
            r"(speech|talking).*(slurred|garbled|difficult)",
            r"sudden.*(vision|blind|blindness)",
            r"sudden.*(confusion|confused|disoriented)",
            r"(stroke signs|FAST|brain attack)",
        ],
        "level": EmergencyLevel.IMMEDIATE,
        "response": "🚨 EMERGENCY: Call 911 immediately! This could be a stroke. " \
                   "Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 911. " \
                   "Note the time symptoms started. Do NOT give aspirin.",
        "category": "neurological"
    },
    "bleeding_emergency": {
        "patterns": [
            r"(severe|heavy|extreme).*(bleeding|blood)",
            r"(can't|cannot|unable to) stop.*(bleeding|blood)",
            r"(blood|bleeding).*(gushing|pulsing|spraying)",
            r"(arterial|artery).*(bleeding|cut)",
            r"(blood loss|losing blood).*(fast|rapid|lot)",
        ],
        "level": EmergencyLevel.IMMEDIATE,
        "response": "🚨 EMERGENCY: Call 911 immediately! Apply firm direct pressure with " \
                   "clean cloth. Do NOT remove cloth if soaked - add more on top. " \
                   "Elevate if possible. Do NOT use tourniquet unless trained.",
        "category": "trauma"
    },
    "overdose_emergency": {
        "patterns": [
            r"(overdose|over-dose|OD'd)",
            r"took too many.*(pill|medication|drug)",
            r"(poison|poisoned|poisoning)",
            r"(drug|medication).*(overdose|over-dose)",
            r"(suicide|suicidal).*(pill|drug|medication)",
        ],
        "level": EmergencyLevel.IMMEDIATE,
        "response": "🚨 EMERGENCY: Call 911 and Poison Control (1-800-222-1222) immediately! " \
                   "Do NOT induce vomiting unless instructed. Keep medication bottle. " \
                   "Stay with the person. Note time and amount taken if known.",
        "category": "toxicology"
    },
    "seizure_emergency": {
        "patterns": [
            r"(seizure|convulsion|fit)",
            r"(shaking|jerking).*(uncontrollable|can't stop)",
            r"(child|baby).*(seizure|convulsion|fever).*(high)",
        ],
        "level": EmergencyLevel.IMMEDIATE,
        "response": "🚨 EMERGENCY: Call 911 if seizure lasts more than 5 minutes or " \
                   "person is injured, pregnant, or has no prior seizure history. " \
                   "Clear area of dangers. Do NOT restrain. Turn on side after jerking stops. " \
                   "Time the seizure. Do NOT put anything in mouth.",
        "category": "neurological"
    },
    
    # URGENT - Go to ER
    "allergic_urgent": {
        "patterns": [
            r"(allergic|allergy).*(reaction|swelling|hives)",
            r"(swelling).*(face|throat|tongue|lips)",
            r"(hives|rash).*(spreading|everywhere|all over)",
            r"(bee|wasp|insect).*(sting|bite).*(allergic|swelling)",
        ],
        "level": EmergencyLevel.URGENT,
        "response": "⚠️ URGENT: Seek emergency care immediately! This could be anaphylaxis. " \
                   "Use EpiPen if prescribed and available. Go to ER even if symptoms improve " \
                   "- they can return. Call 911 if breathing is affected.",
        "category": "allergy"
    },
    "head_injury_urgent": {
        "patterns": [
            r"(head|skull).*(injury|trauma|hit|smash|crack)",
            r"(concussion|brain injury)",
            r"(head).*(bleeding|blood|cut).*(deep|large|gaping)",
        ],
        "level": EmergencyLevel.URGENT,
        "response": "⚠️ URGENT: Go to ER immediately! Watch for: loss of consciousness, " \
                   "confusion, repeated vomiting, unequal pupils, worsening headache. " \
                   "Do NOT let person sleep for first few hours if severe.",
        "category": "trauma"
    },
    "abdominal_urgent": {
        "patterns": [
            r"(severe|extreme).*(abdominal|stomach|belly).*(pain|cramp)",
            r"(abdominal|stomach).*(pain).*(fever|vomit)",
            r"(blood|bleeding).*(stool|vomit|urine)",
        ],
        "level": EmergencyLevel.URGENT,
        "response": "⚠️ URGENT: Go to ER immediately! This could indicate appendicitis, " \
                   "internal bleeding, or other serious condition. Do NOT eat or drink. " \
                   "Note when symptoms started and any other symptoms.",
        "category": "gastrointestinal"
    },
    
    # CONSULT - See doctor soon
    "persistent_symptoms": {
        "patterns": [
            r"(symptom|pain|cough|headache).*(week|weeks|month|months).*(won't|doesn't|not).*(go|stop)",
            r"(persistent|chronic|ongoing).*(pain|symptom|cough|fatigue)",
            r"(losing weight).*(without|no).*(trying|reason)",
        ],
        "level": EmergencyLevel.CONSULT,
        "response": "📋 CONSULT: Please schedule an appointment with your doctor within " \
                   "the next few days. Persistent symptoms should be evaluated. " \
                   "If symptoms worsen before your appointment, go to urgent care or ER.",
        "category": "general"
    },
}


# Blocked phrases - should NEVER appear in medical responses
BLOCKED_RESPONSE_PATTERNS = [
    r"you (definitely|certainly|absolutely) have",
    r"you should (take|start|stop|begin) (this|the) (medication|medicine|drug)",
    r"your diagnosis is",
    r"you are diagnosed with",
    r"(stop|don't|discontinue).*(your|the).*(medication|medicine|treatment)",
    r"take \d+ (mg|milligrams) of",
    r"prescription for you",
]


def apply_rules(query: str) -> RuleMatch:
    """
    Check query against all emergency rules
    
    Returns RuleMatch with appropriate response or no match
    """
    query_lower = query.lower()
    matched_patterns = []
    
    # Check each rule category
    for rule_name, rule_data in EMERGENCY_RULES.items():
        for pattern in rule_data["patterns"]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                matched_patterns.append(pattern)
                
                return RuleMatch(
                    matched=True,
                    level=rule_data["level"],
                    response=rule_data["response"],
                    matched_patterns=matched_patterns,
                    category=rule_data["category"]
                )
    
    return RuleMatch(
        matched=False,
        level=EmergencyLevel.NONE,
        response="",
        matched_patterns=[],
        category=""
    )


def get_emergency_response(query: str) -> Optional[str]:
    """Simple interface - get emergency response or None"""
    match = apply_rules(query)
    return match.response if match.matched else None


def validate_llm_response(query: str, response: str, risk_level: str) -> Tuple[str, bool]:
    """
    Validate and potentially modify LLM response
    
    Returns: (sanitized_response, is_safe)
    """
    is_safe = True
    
    # Check for blocked patterns
    for pattern in BLOCKED_RESPONSE_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            is_safe = False
            # Replace with safer version
            response = re.sub(
                pattern,
                "This requires evaluation by a medical professional to determine.",
                response,
                flags=re.IGNORECASE
            )
    
    # Add disclaimer for medical queries
    if risk_level in ["Critical", "Ambiguous"]:
        disclaimer = "\n\n⚠️ DISCLAIMER: This is not medical advice. " \
                    "Please consult a healthcare professional for proper diagnosis and treatment."
        if disclaimer not in response:
            response += disclaimer
    
    return response, is_safe


def get_rule_categories() -> List[str]:
    """Get list of all emergency categories"""
    return list(set(rule["category"] for rule in EMERGENCY_RULES.values()))


def print_all_rules():
    """Debug: print all rules"""
    for name, rule in EMERGENCY_RULES.items():
        print(f"\n📁 {name}")
        print(f"   Level: {rule['level'].value}")
        print(f"   Category: {rule['category']}")
        print(f"   Patterns: {len(rule['patterns'])}")


if __name__ == "__main__":
    # Test rules
    test_queries = [
        "I have chest pain and can't breathe",
        "My face is drooping and speech is slurred",
        "I have a headache for 3 weeks that won't go away",
        "How do I learn Python",
    ]
    
    for query in test_queries:
        match = apply_rules(query)
        print(f"\nQuery: {query}")
        print(f"Matched: {match.matched}")
        if match.matched:
            print(f"Level: {match.level.value}")
            print(f"Response: {match.response[:100]}...")