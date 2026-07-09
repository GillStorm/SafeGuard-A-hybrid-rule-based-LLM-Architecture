# core/semantic_matcher.py
"""
Semantic Emergency Matching (Level 1)
Uses sentence embeddings + cosine similarity to catch emergency queries
that regex misses (typos, synonyms, paraphrases).

Replaces brittle regex with meaning-based matching.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to load sentence-transformers; graceful fallback if missing
try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Semantic matching disabled.")


# ==========================================
# EMERGENCY CONCEPT BANK
# ==========================================
# Each concept: (description, category, response)
# Descriptions are written in varied natural language so the
# embedding captures MEANING, not just keywords.

EMERGENCY_CONCEPTS = [
    # Cardiac
    {
        "descriptions": [
            "patient experiencing crushing pressure in the chest area",
            "severe tightness or squeezing sensation in the thorax",
            "heart hurting with pain radiating down the arm",
            "intense discomfort in the chest with sweating and nausea",
            "feeling like an elephant is sitting on my chest",
            "sharp stabbing pain in the heart region",
            "my heart feels like its being squeezed",
            "burning sensation in the middle of my chest",
        ],
        "category": "cardiac",
        "response": (
            "🚨 EMERGENCY: Call 911 immediately! This could be a heart attack. "
            "Do NOT drive yourself. Sit or lie down. Chew an aspirin if available "
            "and not allergic. Unlock your door for paramedics."
        ),
    },
    # Respiratory
    {
        "descriptions": [
            "person unable to breathe or gasping for air",
            "severe difficulty getting air into the lungs",
            "choking and cannot get any air",
            "throat closing up and restricting airflow",
            "suffocating and turning blue",
            "airways completely blocked cannot inhale",
            "lungs feel like they are filled with fluid",
            "wheezing so badly I cannot take a breath",
        ],
        "category": "respiratory",
        "response": (
            "🚨 EMERGENCY: Call 911 immediately! Check if person is conscious. "
            "If choking and conscious, perform Heimlich maneuver. "
            "If not breathing, begin CPR if trained."
        ),
    },
    # Stroke
    {
        "descriptions": [
            "signs of a cerebrovascular accident with facial drooping",
            "one side of the face is drooping and arm is weak",
            "speech is suddenly slurred and garbled",
            "sudden loss of vision in one eye",
            "sudden severe confusion and disorientation",
            "half of body suddenly went numb and cannot move",
            "suddenly cannot speak or understand words",
            "worst headache of my life came on suddenly",
        ],
        "category": "neurological",
        "response": (
            "🚨 EMERGENCY: Call 911 immediately! This could be a stroke. "
            "Remember FAST: Face drooping, Arm weakness, Speech difficulty, "
            "Time to call 911. Note the time symptoms started."
        ),
    },
    # Unconsciousness
    {
        "descriptions": [
            "person collapsed and is not responding to anything",
            "found someone unconscious on the ground",
            "patient is unresponsive and will not wake up",
            "someone fainted and has not regained consciousness",
            "person blacked out and is limp",
            "cannot wake someone up no matter what I try",
        ],
        "category": "neurological",
        "response": (
            "🚨 EMERGENCY: Call 911 immediately! Check for breathing and pulse. "
            "If not breathing, start CPR. Place in recovery position if breathing. "
            "Do NOT leave the person alone."
        ),
    },
    # Severe Bleeding
    {
        "descriptions": [
            "massive uncontrollable bleeding from a wound",
            "blood is gushing and spurting from an injury",
            "severed an artery and blood is spraying",
            "deep laceration with heavy blood loss",
            "hemorrhaging badly and losing a lot of blood",
            "wound will not stop bleeding despite pressure",
        ],
        "category": "trauma",
        "response": (
            "🚨 EMERGENCY: Call 911 immediately! Apply firm direct pressure with "
            "clean cloth. Do NOT remove cloth if soaked - add more on top. "
            "Elevate if possible."
        ),
    },
    # Overdose / Poisoning
    {
        "descriptions": [
            "took too many pills and feeling very sick",
            "accidentally ingested poison or toxic substance",
            "drug overdose and becoming unresponsive",
            "swallowed something toxic and feeling faint",
            "intentional self-harm with medication",
            "someone took a whole bottle of pills",
        ],
        "category": "toxicology",
        "response": (
            "🚨 EMERGENCY: Call 911 and Poison Control (1-800-222-1222) immediately! "
            "Do NOT induce vomiting unless instructed. Keep medication bottle. "
            "Stay with the person."
        ),
    },
]


# ==========================================
# SEMANTIC MATCHER
# ==========================================

@dataclass
class SemanticMatch:
    """Result of semantic emergency matching"""
    matched: bool
    similarity: float
    category: str
    response: str
    matched_concept: str  # The description that was closest


# Global model instance (loaded once)
_model = None
_concept_embeddings = None  # List of (embedding, concept_index, description)


def _load_model():
    """Load the sentence transformer model and pre-encode concepts."""
    global _model, _concept_embeddings

    if not ST_AVAILABLE:
        return False

    if _model is not None:
        return True  # Already loaded

    try:
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-encode all concept descriptions
        _concept_embeddings = []
        for idx, concept in enumerate(EMERGENCY_CONCEPTS):
            for desc in concept["descriptions"]:
                embedding = _model.encode(desc, convert_to_tensor=True)
                _concept_embeddings.append((embedding, idx, desc))

        logger.info(
            f"Semantic matcher ready: {len(_concept_embeddings)} concept vectors loaded."
        )
        return True

    except Exception as e:
        logger.error(f"Failed to load semantic matcher: {e}")
        return False


def semantic_match(query: str, threshold: float = 0.55) -> SemanticMatch:
    """
    Match a query against emergency concepts using cosine similarity.

    Args:
        query: The user's input query.
        threshold: Minimum similarity score to consider a match (0-1).

    Returns:
        SemanticMatch with match details or no-match.
    """
    if not _load_model():
        return SemanticMatch(
            matched=False, similarity=0.0,
            category="", response="", matched_concept=""
        )

    # Encode the query
    query_embedding = _model.encode(query, convert_to_tensor=True)

    best_score = 0.0
    best_concept_idx = -1
    best_desc = ""

    # Compare against every pre-encoded concept description
    for emb, concept_idx, desc in _concept_embeddings:
        score = util.cos_sim(query_embedding, emb).item()
        if score > best_score:
            best_score = score
            best_concept_idx = concept_idx
            best_desc = desc

    if best_score >= threshold and best_concept_idx >= 0:
        concept = EMERGENCY_CONCEPTS[best_concept_idx]
        return SemanticMatch(
            matched=True,
            similarity=best_score,
            category=concept["category"],
            response=concept["response"],
            matched_concept=best_desc,
        )

    return SemanticMatch(
        matched=False, similarity=best_score,
        category="", response="", matched_concept=""
    )


def get_all_similarities(query: str) -> List[Tuple[str, float, str]]:
    """
    Debug helper: get similarity scores for all concepts.
    Returns list of (description, score, category) sorted by score desc.
    """
    if not _load_model():
        return []

    query_embedding = _model.encode(query, convert_to_tensor=True)
    results = []

    for emb, concept_idx, desc in _concept_embeddings:
        score = util.cos_sim(query_embedding, emb).item()
        category = EMERGENCY_CONCEPTS[concept_idx]["category"]
        results.append((desc, score, category))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


if __name__ == "__main__":
    print("Testing Semantic Emergency Matcher\n")
    print("=" * 60)

    test_queries = [
        # Should match (paraphrases / typos / synonyms)
        "pain in my thorax",
        "my heart hurts really bad",
        "chesst pain and sweating",
        "I feel like something is squeezing my chest",
        "cant get any air into my lungs",
        "one side of face is drooping",
        "someone collapsed and wont wake up",
        # Should NOT match
        "I have a mild headache",
        "how do I learn Python",
        "what is diabetes",
    ]

    for q in test_queries:
        result = semantic_match(q)
        status = "✅ MATCH" if result.matched else "❌ NO MATCH"
        print(f"\nQuery: {q}")
        print(f"  {status} (similarity: {result.similarity:.3f})")
        if result.matched:
            print(f"  Category: {result.category}")
            print(f"  Closest: {result.matched_concept}")
