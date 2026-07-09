# core/rules.py
"""
Ultra-Fast Hybrid Safety Rules Engine
Layer 1: Regex (< 1ms)
Layer 2: LLM Semantic Judge via Groq (~30ms)
NO LOCAL MODEL LOADING.
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
import os

# Import your existing LLM client from llm.py
from core.llm import client, MODEL_NAME

logger = logging.getLogger(__name__)

class EmergencyLevel(Enum):
    IMMEDIATE = "immediate"
    NONE = "none"

@dataclass
class RuleMatch:
    matched: bool
    level: EmergencyLevel
    response: str
    category: str
    match_type: str 

# --- LAYER 1: REGEX ---
REGEX_RULES = {
    "cardiac_regex": {
        "patterns": [
            r"(crushing|severe).*(chest|heart)",
            r"(chest).*(pain).*(arm|jaw|back)",
            r"heart attack",
            r"cardiac arrest",
        ],
        "response": "🚨 EMERGENCY: Call 911 immediately! This could be a heart attack. Do NOT drive yourself. Sit or lie down. Chew an aspirin if available.",
        "category": "cardiac"
    },
    "breathing_regex": {
        "patterns": [r"(can'?t|cannot).*(breathe|breathing)", r"(throat).*(closing|swelling)"],
        "response": "🚨 EMERGENCY: Call 911 immediately! Check airway. If choking, perform Heimlich. If not breathing, begin CPR.",
        "category": "respiratory"
    },
    "stroke_regex": {
        "patterns": [r"(face|arm).*(droop|numb|weak)", r"(slurred).*(speech)"],
        "response": "🚨 EMERGENCY: Call 911 immediately! This could be a stroke. Note the time symptoms started.",
        "category": "neurological"
    }
}

SEMANTIC_RESPONSES = {
    "cardiac": "🚨 EMERGENCY: Based on your description, this sounds like a cardiac event. Call 911 immediately. Do not drive.",
    "stroke": "🚨 EMERGENCY: Based on your description, this sounds like a neurological emergency. Call 911 immediately.",
    "breathing": "🚨 EMERGENCY: Based on your description, this sounds like a respiratory emergency. Call 911 immediately."
}

def _check_llm_semantic(query: str) -> dict:
    """
    LLM-as-a-Judge: Asks Groq/Llama if the query implies an emergency.
    Takes ~30ms. No local downloading.
    """
    if not client:
        return None
        
    # Extremely strict, fast prompt for Llama to classify
    prompt = f"""Analyze this user text for a medical safeguard system.
Text: "{query}"
Does this text actually describe a genuine medical emergency (e.g., heart attack, stroke, severe bleeding, unable to breathe)?
CRITICAL:
1. Watch out for common medical idioms indicating severe emergencies (e.g., "elephant sitting on my chest", "elephant on my ribcage"). These ARE emergencies.
2. Pay close attention to context. If the symptoms have a clear, harmless external cause (like visiting a dentist, sleeping in a weird position, or working out), it is NOT an emergency.
Answer ONLY with one word for the category (CARDIAC, STROKE, BREATHING, TRAUMA) or NONE."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,      # Zero creativity, pure logic
            max_tokens=5,         # We only want one word back
            timeout=2.0           # Fail fast if Groq is down
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        # Map LLM response to our categories
        if "CARDIAC" in result:
            return {"category": "cardiac", "confidence": 0.9}
        elif "STROKE" in result:
            return {"category": "stroke", "confidence": 0.9}
        elif "BREATH" in result:
            return {"category": "breathing", "confidence": 0.9}
        elif "TRAUMA" in result:
            return {"category": "trauma", "confidence": 0.9}
            
    except Exception as e:
        logger.error(f"LLM Semantic check failed: {e}")
        
    return None

def apply_rules(query: str) -> RuleMatch:
    """Hybrid Rule Application - Instantaneous"""
    
    # LAYER 1: Regex (Takes < 1 millisecond)
    for rule_name, rule_data in REGEX_RULES.items():
        for pattern in rule_data["patterns"]:
            if re.search(pattern, query, re.IGNORECASE):
                return RuleMatch(
                    matched=True,
                    level=EmergencyLevel.IMMEDIATE,
                    response=rule_data["response"],
                    category=rule_data["category"],
                    match_type="regex"
                )
                
    # LAYER 2: LLM Semantic Check (Takes ~30 milliseconds via Groq API)
    # ONLY triggers if Regex fails, to save API calls
    semantic_hit = _check_llm_semantic(query)
    if semantic_hit:
        category = semantic_hit["category"]
        return RuleMatch(
            matched=True,
            level=EmergencyLevel.IMMEDIATE,
            response=SEMANTIC_RESPONSES.get(category, "🚨 EMERGENCY detected via AI."),
            category=category,
            match_type="llm-semantic" 
        )
        
    return RuleMatch(matched=False, level=EmergencyLevel.NONE, response="", category="", match_type="none")


# Blocked phrases - should NEVER appear in LLM responses
BLOCKED_RESPONSE_PATTERNS = [
    r"you (definitely|certainly|absolutely) have",
    r"you should (take|start|stop|begin) (this|the) (medication|medicine|drug)",
    r"your diagnosis is",
    r"you are diagnosed with",
    r"take \d+ (mg|milligrams) of",
]

def validate_llm_response(query: str, response: str, risk_level: str) -> tuple:
    """Validate and sanitize LLM response"""
    is_safe = True
    
    for pattern in BLOCKED_RESPONSE_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            is_safe = False
            response = re.sub(
                pattern,
                "This requires evaluation by a medical professional.",
                response,
                flags=re.IGNORECASE
            )
    
    # Add disclaimer for medical queries
    if risk_level in ["Critical", "Ambiguous"]:
        disclaimer = "\n\n⚠️ DISCLAIMER: This is not medical advice. " \
                    "Please consult a healthcare professional."
        if disclaimer not in response:
            response += disclaimer
    
    return response, is_safe