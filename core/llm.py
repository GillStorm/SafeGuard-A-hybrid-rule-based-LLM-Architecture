# core/llm.py
"""
LLM Integration Module - Groq (Llama 3.1)
Uses Groq API for ultra-fast Llama inference with OpenAI-compatible SDK.
"""

import os
import logging
from typing import List, Optional

try:
    # pyrefly: ignore [missing-import]
    from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
# Groq uses the exact same format as OpenAI, but points to their servers
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant"  # Fast, lightweight Llama 3.1 model on Groq

# System prompt specifically tuned for medical safety
MEDICAL_SYSTEM_PROMPT = """You are a helpful medical information assistant integrated into a safeguard system. 

CRITICAL SAFETY RULES:
1. NEVER provide a definitive medical diagnosis.
2. NEVER recommend specific prescription medications or exact dosages.
3. ALWAYS clarify that you are an AI, not a doctor.
4. If the user describes severe symptoms (chest pain, breathing issues, bleeding), strongly emphasize seeking immediate emergency care.
5. Keep responses concise, empathetic, and factual.
"""

# Safe fallback if API fails, times out, or hits rate limits
SAFE_FALLBACK_RESPONSE = (
    "I apologize, but I am currently unable to generate a detailed response. "
    "For any medical concerns, please consult a healthcare professional or call "
    "emergency services if you are experiencing a medical emergency."
)

# ==========================================
# CLIENT INITIALIZATION
# ==========================================
client = None

def _init_client():
    """Initialize Groq client safely using the OpenAI SDK."""
    global client
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI library not installed. Run: pip install openai")
        return False
        
    # Using GROQ_API_KEY instead of OPENAI_API_KEY
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment variables.")
        return False
        
    try:
        # Point the OpenAI client to Groq's base URL
        client = OpenAI(
            api_key=api_key, 
            base_url=GROQ_BASE_URL
        )
        logger.info("Groq (Llama) client initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return False

# Attempt initialization on module load
GROQ_READY = _init_client()


# ==========================================
# MAIN LLM RESPONSE FUNCTION
# ==========================================
def llm_response(query: str, conversation_history: Optional[List[dict]] = None) -> str:
    """
    Generate Llama response via Groq with safety fallbacks.
    
    Args:
        query: The user's input query.
        conversation_history: Optional list of previous messages.
        
    Returns:
        str: The LLM response or a safe fallback message.
    """
    if not GROQ_READY:
        # If no API key is set, return the placeholder to prevent crashes
        return f"[Placeholder] I understand you are asking about: '{query}'. Please consult a medical professional."

    messages = [{"role": "system", "content": MEDICAL_SYSTEM_PROMPT}]
    
    if conversation_history:
        messages.extend(conversation_history)
        
    messages.append({"role": "user", "content": query})
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,   # Low temp for factual medical info
            max_tokens=500,    # Prevent long hallucinations
            timeout=10.0       # Fail fast for safeguard systems
        )
        return response.choices[0].message.content
        
    except AuthenticationError:
        logger.error("Invalid Groq API Key.")
        return SAFE_FALLBACK_RESPONSE
        
    except RateLimitError:
        logger.warning("Groq Rate Limit hit. Using fallback.")
        return SAFE_FALLBACK_RESPONSE
        
    except (APIConnectionError, APITimeoutError) as e:
        logger.warning(f"Groq Connection Error: {e}. Using fallback.")
        return SAFE_FALLBACK_RESPONSE
        
    except Exception as e:
        logger.error(f"Unexpected Groq Error: {e}")
        return SAFE_FALLBACK_RESPONSE


if __name__ == "__main__":
    # Quick test of the LLM module
    print("Testing LLM Module...")
    print(f"OpenAI SDK Available: {OPENAI_AVAILABLE}")
    print(f"Groq Ready: {GROQ_READY}")
    print(f"Model: {MODEL_NAME}")
    
    test_query = "What are the common symptoms of a cold?"
    
    print(f"\nQuery: {test_query}")
    response = llm_response(test_query)
    print(f"Response: {response}\n")