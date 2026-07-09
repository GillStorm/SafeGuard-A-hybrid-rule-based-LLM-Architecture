# core/pipeline.py
"""
Medical Safeguard LLM Pipeline
Complete safe response generation
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from core.classifier import classify
from core.rules import apply_rules, validate_llm_response
from core.risk import compute_advanced_risk, RiskCategory
from core.llm import llm_response
from core.session_state import SessionState, update_session
from core.llm_judge import verify_with_judge


# Configuration
THRESHOLD = 0.3
SYSTEM_PROMPT = """You are a medical information assistant. You MUST follow these rules:

1. NEVER provide a diagnosis
2. NEVER recommend specific medications or dosages
3. ALWAYS include a disclaimer that this is not medical advice
4. For any concerning symptoms, recommend consulting a healthcare professional
5. If the query seems urgent, explicitly recommend seeking immediate medical care
6. Keep responses factual and evidence-based
7. When unsure, err on the side of caution

Your response format:
- Acknowledge the concern
- Provide general, safe information
- Include appropriate disclaimer
- Recommend professional consultation when appropriate
"""


@dataclass
class PipelineResult:
    """Complete pipeline result"""
    query: str
    response: str
    label: str
    confidence: float
    risk_score: float
    risk_category: str
    rule_triggered: bool
    response_modified: bool
    judge_score: int
    judge_safe: bool
    response_time_ms: float
    timestamp: str


def safe_guard(query: str, 
               conversation_history: List[str] = None,
               session_state: SessionState = None) -> PipelineResult:
    """
    Main safeguard pipeline
    
    Flow:
    1. Classify query
    2. Update Conversational Session State
    3. Calculate risk
    4. Check emergency rules
    5. Generate/block LLM response
    6. Validate response (Regex + LLM Judge)
    """
    start_time = datetime.now()
    
    # Step 1: Classification
    classification = classify(query)
    label = classification["label"]
    confidence = classification["severity"]
    
    # Step 2: Update Conversational Session State
    session_escalation = None
    if session_state is not None:
        session_update = update_session(query, classification, session_state)
        if session_update.escalation_triggered:
            session_escalation = session_update.escalation_response

    # Step 3: Advanced risk calculation
    risk_assessment = compute_advanced_risk(
        query, label, confidence, conversation_history
    )
    
    # Step 4: Check emergency rules
    rule_match = apply_rules(query)
    
    # Determine response path
    rule_triggered = False
    response_modified = False
    judge_score = 10
    judge_safe = True
    
    if session_escalation:
        # Conversational symptom accumulation triggered an emergency
        response = session_escalation
        rule_triggered = True
        
    elif rule_match.matched:
        # Emergency rule matched - use rule response
        response = rule_match.response
        rule_triggered = True
        
    elif risk_assessment.should_block_llm:
        # High risk but no specific rule - use safe fallback
        response = _get_safe_fallback(label, risk_assessment.category)
        
    else:
        # Safe to use LLM
        raw_response = llm_response(query)
        
        # Step 6a: Regex validation
        response, is_safe_regex = validate_llm_response(
            query, raw_response, label
        )
        
        # Step 6b: LLM Judge Verification
        judge_verdict = verify_with_judge(query, response)
        judge_score = judge_verdict.score
        judge_safe = judge_verdict.safe
        
        if not judge_safe:
            # Overwrite with safe fallback
            response = "⚠️ This response was flagged by the medical safety auditor and has been removed. Please consult a healthcare professional."
            is_safe_regex = False
        
        if not is_safe_regex or not judge_safe:
            response_modified = True
    
    # Calculate response time
    response_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return PipelineResult(
        query=query,
        response=response,
        label=label,
        confidence=confidence,
        risk_score=risk_assessment.overall_risk,
        risk_category=risk_assessment.category.value,
        rule_triggered=rule_triggered,
        response_modified=response_modified,
        judge_score=judge_score,
        judge_safe=judge_safe,
        response_time_ms=response_time,
        timestamp=datetime.now().isoformat()
    )


def _get_safe_fallback(label: str, risk_category: RiskCategory) -> str:
    """Generate safe fallback response when LLM is blocked"""
    
    if risk_category == RiskCategory.CRITICAL or label == "Critical":
        return ("🚨 Based on your description, this requires immediate medical attention. "
                "Please call emergency services (911) or go to your nearest emergency room. "
                "Do not wait - some conditions worsen rapidly without treatment.")
    
    elif risk_category == RiskCategory.HIGH:
        return ("⚠️ Based on your description, you should seek medical care soon. "
                "Please contact your doctor or visit urgent care today. "
                "If symptoms worsen, go to the emergency room.")
    
    elif risk_category == RiskCategory.MEDIUM:
        return ("📋 Based on your description, I recommend consulting with a healthcare "
                "professional for proper evaluation. While this may not be an emergency, "
                "a doctor can provide accurate diagnosis and treatment options.")
    
    else:
        return ("For accurate medical information, please consult a healthcare professional. "
                "They can provide personalized advice based on your complete medical history.")


def safe_guard_stream(query: str):
    """
    Streaming version for real-time applications
    Yields partial results as they're computed
    """
    import json
    
    # Yield classification result
    classification = classify(query)
    label = classification["label"]
    confidence = classification["severity"]
    yield json.dumps({
        "stage": "classification",
        "label": label,
        "confidence": confidence
    })
    
    # Yield risk assessment
    risk = compute_advanced_risk(query, label, confidence)
    yield json.dumps({
        "stage": "risk_assessment",
        "risk_score": risk.overall_risk,
        "risk_category": risk.category.value,
        "should_block": risk.should_block_llm
    })
    
    # Yield final response
    result = safe_guard(query)
    yield json.dumps({
        "stage": "response",
        "response": result.response,
        "rule_triggered": result.rule_triggered
    })


if __name__ == "__main__":
    # Test pipeline
    test_queries = [
        "I have chest pain and can't breathe",
        "I feel dizzy sometimes",
        "How do I learn Python",
        "I have pain in my left arm, started 2 hours ago, getting worse",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("-"*60)
        
        result = safe_guard(query)
        
        print(f"Label: {result.label}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Risk: {result.risk_score:.2f} ({result.risk_category})")
        print(f"Rule Triggered: {result.rule_triggered}")
        print(f"Response Modified: {result.response_modified}")
        print(f"Response Time: {result.response_time_ms:.1f}ms")
        print("-"*60)
        print(f"Response: {result.response}")