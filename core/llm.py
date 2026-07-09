# core/llm.py
"""
LLM Integration Module - Groq (Llama 3.1)
Uses Groq API for ultra-fast Llama inference with OpenAI-compatible SDK.
"""

import os
import logging
from typing import List, Optional

try:
    from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant"  # Used by both llm.py AND rules.py

MEDICAL_SYSTEM_PROMPT = """You are a helpful medical information assistant integrated into a safeguard system. 

CRITICAL SAFETY RULES:
1. NEVER provide a definitive medical diagnosis.
2. NEVER recommend specific prescription medications or exact dosages.
3. ALWAYS clarify that you are an AI, not a doctor.
4. If the user describes severe symptoms (chest pain, breathing issues, bleeding), strongly emphasize seeking immediate emergency care.
5. Keep responses concise, empathetic, and factual.
"""

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
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment variables.")
        return False
        
    try:
        # FIX: Timeout goes HERE in the client initialization, not in .create()
        client = OpenAI(
            api_key=api_key, 
            base_url=GROQ_BASE_URL,
            timeout=10.0  # <--- MOVED HERE
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
    """Generate Llama response via Groq with safety fallbacks."""
    if not GROQ_READY:
        return f"[Placeholder] I understand you are asking about: '{query}'. Please consult a medical professional."

    messages = [{"role": "system", "content": MEDICAL_SYSTEM_PROMPT}]
    
    if conversation_history:
        messages.extend(conversation_history)
        
    messages.append({"role": "user", "content": query})
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,   
            max_tokens=500,    
            # FIX: Removed timeout=10.0 from here
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