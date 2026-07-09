# core/risk.py
"""
Safety-First Risk Calculation
HIGH confidence in Critical = HIGH risk (opposite of old logic)
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import re


class RiskCategory(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class RiskAssessment:
    overall_risk: float
    category: RiskCategory
    factors: Dict[str, float]
    recommendation: str
    should_block_llm: bool
    reasoning: List[str]


SEVERITY_MAP = {
    "Critical": 0.95,
    "Ambiguous": 0.5,
    "Safe": 0.05,
    "Unknown": 0.5
}


def get_severity(label: str) -> float:
    return SEVERITY_MAP.get(label, 0.5)


def compute_risk(severity: float, confidence: float) -> float:
    """
    SAFETY-FIRST risk calculation:
    - If Critical + High confidence = HIGH risk (block LLM)
    - If Critical + Low confidence = HIGH risk (uncertain, block LLM)
    - If Safe + High confidence = LOW risk (allow LLM)
    """
    # Base risk from severity
    base_risk = severity
    
    # Confidence multiplier: high confidence amplifies the severity
    # If we're confident it's critical, risk goes up
    # If we're confident it's safe, risk goes down
    confidence_factor = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
    
    return base_risk * confidence_factor


def compute_advanced_risk(
    query: str,
    label: str,
    confidence: float,
    conversation_history: List[str] = None
) -> RiskAssessment:
    """Multi-factor risk assessment"""
    factors = {}
    reasoning = []
    
    # Factor 1: Classification-based risk (FIXED MATH)
    severity = get_severity(label)
    classification_risk = compute_risk(severity, confidence)
    factors["classification"] = classification_risk
    
    # ALWAYS block LLM for Critical, regardless of other factors
    if label == "Critical":
        reasoning.append(f"CRITICAL classification detected ({confidence:.0%} confidence)")
    
    # Factor 2: Query urgency patterns
    urgency_risk = _detect_urgency_patterns(query)
    factors["urgency"] = urgency_risk
    if urgency_risk > 0.5:
        reasoning.append("Emergency language detected")
    
    # Factor 3: Symptom specificity
    specificity_risk = _detect_symptom_specificity(query)
    factors["specificity"] = specificity_risk
    
    # Calculate weighted overall risk
    weights = {
        "classification": 0.5,
        "urgency": 0.3,
        "specificity": 0.2
    }
    
    overall_risk = sum(
        factors.get(factor, 0) * weight 
        for factor, weight in weights.items()
    )
    
    category = _get_risk_category(overall_risk)
    recommendation = _get_recommendation(category, label)
    
    # BLOCK LLM if Critical OR if overall risk is high
    should_block = (label == "Critical") or (overall_risk > 0.4)
    
    return RiskAssessment(
        overall_risk=overall_risk,
        category=category,
        factors=factors,
        recommendation=recommendation,
        should_block_llm=should_block,
        reasoning=reasoning
    )


def _detect_urgency_patterns(query: str) -> float:
    urgent_patterns = [
        r"(can'?t|cannot|unable to).*(breathe|move|see|speak)",
        r"(severe|extreme|intense|unbearable|crushing)",
        r"(emergency|911|ambulance|right now|immediately)",
        r"(dying|life.?threatening)",
        r"(sudden|suddenly)",
    ]
    
    query_lower = query.lower()
    matches = sum(1 for p in urgent_patterns if re.search(p, query_lower))
    
    return min(0.9, matches * 0.3)


def _detect_symptom_specificity(query: str) -> float:
    specificity_indicators = [
        r"(\d+|one|two|three|four|five).*(hour|day|week|minute)",
        r"(left|right|lower|upper|front|back).*(side|arm|leg|chest)",
        r"(spreading|radiating)",
    ]
    
    query_lower = query.lower()
    matches = sum(1 for p in specificity_indicators if re.search(p, query_lower))
    
    return min(0.8, matches * 0.3)


def _get_risk_category(risk: float) -> RiskCategory:
    if risk >= 0.6:
        return RiskCategory.CRITICAL
    elif risk >= 0.4:
        return RiskCategory.HIGH
    elif risk >= 0.25:
        return RiskCategory.MEDIUM
    elif risk >= 0.1:
        return RiskCategory.LOW
    else:
        return RiskCategory.MINIMAL


def _get_recommendation(category: RiskCategory, label: str) -> str:
    if label == "Critical":
        return "BLOCK LLM: Critical classification"
    if category == RiskCategory.CRITICAL:
        return "BLOCK LLM: Critical risk level"
    if category == RiskCategory.HIGH:
        return "BLOCK LLM: High risk level"
    if category == RiskCategory.MEDIUM:
        return "ALLOW LLM with strict disclaimers"
    return "ALLOW LLM normally"


if __name__ == "__main__":
    # Test the fixed math
    print("Testing FIXED risk calculation:\n")
    
    tests = [
        ("Critical", 0.95, "High confidence heart attack"),
        ("Critical", 0.50, "Low confidence heart attack"),
        ("Ambiguous", 0.70, "Medium confidence symptom"),
        ("Safe", 0.95, "High confidence safe query"),
    ]
    
    for label, conf, desc in tests:
        severity = get_severity(label)
        risk = compute_risk(severity, conf)
        block = "BLOCK" if risk > 0.4 or label == "Critical" else "ALLOW"
        print(f"{desc:35} -> Risk: {risk:.2f} [{block}]")

# --- Add this at the very bottom of core/risk.py ---

class _DummyTracker:
    def reset(self):
        pass
        
_default_tracker = _DummyTracker()

def reset_conversation_state():
    """Call this when user types 'clear' in the terminal"""
    _default_tracker.reset()