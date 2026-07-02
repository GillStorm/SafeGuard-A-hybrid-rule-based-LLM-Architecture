# core/risk.py
"""
Advanced Risk Calculation System
Multi-factor risk assessment for medical queries
"""

from typing import Dict, List, Tuple
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
    """Complete risk assessment result"""
    overall_risk: float
    category: RiskCategory
    factors: Dict[str, float]
    recommendation: str
    should_block_llm: bool
    reasoning: List[str]


# Severity mappings
SEVERITY_MAP = {
    "Critical": 0.9,
    "Ambiguous": 0.5,
    "Safe": 0.1,
    "Unknown": 0.5  # Treat unknown as moderate risk
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3
}


def get_severity(label: str) -> float:
    """Get severity score for classification label"""
    return SEVERITY_MAP.get(label, 0.5)


def compute_risk(severity: float, confidence: float) -> float:
    """
    Base risk calculation
    Higher severity + lower confidence = higher risk
    """
    return severity * (1 - confidence)


def compute_advanced_risk(
    query: str,
    label: str,
    confidence: float,
    conversation_history: List[str] = None
) -> RiskAssessment:
    """
    Advanced multi-factor risk assessment
    
    Factors:
    1. Classification severity
    2. Classification confidence
    3. Query urgency patterns
    4. Medical terminology density
    5. Conversation escalation (if history provided)
    """
    factors = {}
    reasoning = []
    
    # Factor 1: Classification-based risk
    severity = get_severity(label)
    classification_risk = compute_risk(severity, confidence)
    factors["classification"] = classification_risk
    
    if classification_risk > 0.4:
        reasoning.append(f"High classification risk: {label} with {confidence:.0%} confidence")
    
    # Factor 2: Query urgency patterns
    urgency_risk = _detect_urgency_patterns(query)
    factors["urgency"] = urgency_risk
    
    if urgency_risk > 0.5:
        reasoning.append("Emergency keywords detected in query")
    
    # Factor 3: Medical terminology density
    medical_risk = _detect_medical_terminology(query)
    factors["medical_terminology"] = medical_risk
    
    if medical_risk > 0.6:
        reasoning.append("Complex medical terminology detected")
    
    # Factor 4: Conversation escalation (optional)
    escalation_risk = 0.0
    if conversation_history:
        escalation_risk = _detect_escalation(conversation_history)
        factors["escalation"] = escalation_risk
        
        if escalation_risk > 0.5:
            reasoning.append("Conversation shows symptom escalation")
    
    # Factor 5: Specificity of symptoms
    specificity_risk = _detect_symptom_specificity(query)
    factors["specificity"] = specificity_risk
    
    # Calculate weighted overall risk
    weights = {
        "classification": 0.4,
        "urgency": 0.25,
        "medical_terminology": 0.1,
        "escalation": 0.15,
        "specificity": 0.1
    }
    
    overall_risk = sum(
        factors.get(factor, 0) * weight 
        for factor, weight in weights.items()
    )
    
    # Determine category
    category = _get_risk_category(overall_risk)
    
    # Generate recommendation
    recommendation = _get_recommendation(category, label)
    
    # Should block LLM?
    should_block = overall_risk > 0.4 or label == "Critical"
    
    return RiskAssessment(
        overall_risk=overall_risk,
        category=category,
        factors=factors,
        recommendation=recommendation,
        should_block_llm=should_block,
        reasoning=reasoning
    )


def _detect_urgency_patterns(query: str) -> float:
    """Detect urgency indicators in query"""
    urgent_patterns = {
        "immediate": [
            r"(can't|cannot|unable to).*(breathe|move|see|speak)",
            r"(severe|extreme|intense|unbearable)",
            r"(emergency|emergency room|911|ambulance)",
            r"(right now|immediately|urgently|help)",
            r"(dying|going to die|life.?threatening)",
        ],
        "high": [
            r"(sudden|suddenly|all of a sudden)",
            r"(getting worse|worse|worsening|increasing)",
            r"(very|really|so).*(bad|painful|scared|worried)",
            r"(hour|hours|minutes).*(ago|now)",
        ]
    }
    
    query_lower = query.lower()
    score = 0.0
    
    for pattern in urgent_patterns["immediate"]:
        if re.search(pattern, query_lower):
            score = max(score, 0.9)
            break
    
    if score < 0.9:
        for pattern in urgent_patterns["high"]:
            if re.search(pattern, query_lower):
                score = max(score, 0.6)
    
    return score


def _detect_medical_terminology(query: str) -> float:
    """Detect presence of medical terminology"""
    medical_terms = [
        "diagnosis", "symptom", "prognosis", "treatment",
        "medication", "prescription", "dosage", "contraindication",
        "chronic", "acute", "benign", "malignant",
        "hypertension", "hypotension", "tachycardia", "bradycardia",
        "myocardial", "pulmonary", "cardiac", "neurological",
        "hemorrhage", "thrombosis", "embolism", "ischemia",
        "arrhythmia", "palpitation", "edema", "cyanosis"
    ]
    
    query_lower = query.lower()
    count = sum(1 for term in medical_terms if term in query_lower)
    
    # Scale: 0 terms = 0, 1 term = 0.5, 2+ terms = 0.8
    if count == 0:
        return 0.0
    elif count == 1:
        return 0.5
    else:
        return min(0.8, 0.5 + count * 0.1)


def _detect_escalation(history: List[str]) -> float:
    """Detect if conversation shows symptom escalation"""
    if len(history) < 2:
        return 0.0
    
    escalation_indicators = [
        "getting worse",
        "worse",
        "more",
        "spreading",
        "increasing",
        "now I",
        "also",
        "and now"
    ]
    
    recent_messages = history[-3:]
    score = 0.0
    
    for msg in recent_messages:
        msg_lower = msg.lower()
        for indicator in escalation_indicators:
            if indicator in msg_lower:
                score = min(0.9, score + 0.3)
    
    return score


def _detect_symptom_specificity(query: str) -> float:
    """More specific symptoms = potentially higher risk"""
    specificity_indicators = [
        # Time-specific
        r"(\d+|one|two|three|four|five).*(hour|day|week|minute)",
        # Location-specific
        r"(left|right|lower|upper|front|back).*(side|arm|leg|chest)",
        # Intensity-specific
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten).*(out of|/10|/ 10)",
        # Frequency-specific
        r"(\d+|several|multiple|frequent|constant|continuous)",
    ]
    
    query_lower = query.lower()
    count = sum(1 for pattern in specificity_indicators 
                if re.search(pattern, query_lower))
    
    return min(0.8, count * 0.3)


def _get_risk_category(risk: float) -> RiskCategory:
    """Map risk score to category"""
    if risk >= 0.7:
        return RiskCategory.CRITICAL
    elif risk >= 0.5:
        return RiskCategory.HIGH
    elif risk >= 0.3:
        return RiskCategory.MEDIUM
    elif risk >= 0.15:
        return RiskCategory.LOW
    else:
        return RiskCategory.MINIMAL


def _get_recommendation(category: RiskCategory, label: str) -> str:
    """Get recommendation based on risk category"""
    recommendations = {
        RiskCategory.CRITICAL: "Block LLM. Apply emergency rules immediately.",
        RiskCategory.HIGH: "Block LLM. Provide urgent care guidance.",
        RiskCategory.MEDIUM: "Allow LLM with strict medical disclaimers.",
        RiskCategory.LOW: "Allow LLM with general disclaimer.",
        RiskCategory.MINIMAL: "Allow LLM response normally."
    }
    
    # Override for Critical label
    if label == "Critical":
        return "Block LLM. Critical classification detected."
    
    return recommendations[category]


def format_risk_report(assessment: RiskAssessment) -> str:
    """Format risk assessment for display"""
    lines = [
        f"Risk Score: {assessment.overall_risk:.2f} ({assessment.category.value})",
        "\nRisk Factors:"
    ]
    
    for factor, value in assessment.factors.items():
        bar = "█" * int(value * 20)
        lines.append(f"  {factor:20} {value:.2f} {bar}")
    
    if assessment.reasoning:
        lines.append("\nReasoning:")
        for reason in assessment.reasoning:
            lines.append(f"  • {reason}")
    
    lines.append(f"\nRecommendation: {assessment.recommendation}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test risk calculation
    test_cases = [
        ("I have severe chest pain and can't breathe", "Critical", 0.95),
        ("I feel dizzy sometimes", "Ambiguous", 0.6),
        ("How do I learn Python", "Safe", 0.99),
        ("I have pain in my left arm that started 2 hours ago and is getting worse", "Ambiguous", 0.5),
    ]
    
    for query, label, conf in test_cases:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        assessment = compute_advanced_risk(query, label, conf)
        print(format_risk_report(assessment))