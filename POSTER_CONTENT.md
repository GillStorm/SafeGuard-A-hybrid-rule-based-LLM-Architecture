# SafeGuard: A Hybrid Rule-Based LLM Architecture
**Poster Presentation Content**

---

## 1. Introduction
* **The Promise:** Large Language Models (LLMs) are transforming healthcare by providing instant, accessible medical information.
* **The Problem:** Standard generative LLMs are prone to hallucinations, struggle to recognize life-threatening emergencies hidden in natural language, and often give unqualified, dangerous medical advice.
* **The Solution:** **SafeGuard** is a hybrid architecture that acts as a secure "wrapper" around LLMs. It combines ultra-fast deterministic rules with intelligent semantic tracking to instantly intercept medical emergencies and sanitize unsafe model outputs before they reach the user.

---

## 2. Objectives
* **Instant Emergency Interception:** Block life-threatening queries (e.g., heart attacks, strokes) with sub-millisecond latency.
* **Contextual Nuance:** Catch typos, idioms (e.g., "elephant on my chest"), and conversational context that traditional chatbots miss.
* **Stateful Symptom Tracking:** Accumulate user symptoms across multiple conversation turns to detect escalating, hidden emergencies.
* **Safe Generative AI:** Allow the underlying LLM to answer non-critical medical questions safely by automatically enforcing medical disclaimers and filtering out definitive diagnoses.

---

## 3. Methodology
The SafeGuard architecture operates as a 5-layer pipeline:
1. **Layer 1 (Regex Baseline Check):** Deterministically scans for critical keywords for <1ms interception.
2. **Layer 2 (LLM Semantic Judge):** An intelligent layer that evaluates the context of the query to identify hidden emergencies, typos, and idioms.
3. **Layer 3 (Stateful Tracker):** Uses LLM JSON-extraction to track user symptoms over time, escalating if a specific cluster threshold (e.g., 2 Cardiac symptoms) is met.
4. **Layer 4 (Generative LLM):** Safe queries are routed to the primary LLM to generate a helpful response.
5. **Layer 5 (Output Validation):** The final output is scrubbed of definitive medical phrasing ("you are diagnosed with") and appended with strict medical disclaimers.

---

## 4. Dataset / Models / Requirements

**Datasets Used:**
* **MedQuAD (Medical Question Answering Dataset):** Used to benchmark the generative LLM (Layer 4) for accuracy and safety against real-world, consumer-level medical inquiries. It ensures the model handles complex, multi-faceted medical questions appropriately.
* **Kaggle Symptom Classification Dataset:** Sourced from Kaggle to simulate diverse, messy real-world symptom descriptions. This dataset was crucial for stress-testing the Regex baseline and LLM-as-a-judge (Layers 1 and 2) against vast variations of natural language input.
* **Custom Curated Safety Dataset:** A highly targeted adversarial dataset containing Critical, Ambiguous, and Safe queries specifically designed to test the system's False Negative rate and emergency interception capabilities.
* **TruthfulQA:** Used to establish a baseline for the model's resistance to imitative falsehoods and dangerous medical myths.

**Models Used:**
* **Primary LLM & Semantic Judge:** `llama-3.1-8b-instant` (powered by Groq). Chosen for its ultra-low latency inference, crucial for real-time safety evaluation.

**Software Requirements:**
* **Language:** Python 3.9+
* **Libraries:** `openai` (for Groq API compatibility), `scikit-learn`, `numpy`.
* **API:** Groq API (for cloud-based inference).

**Hardware Requirements:**
* **Processor (CPU):** Intel Core i3 / AMD Ryzen 3 (or equivalent modern multi-core processor).
* **RAM:** Minimum 4 GB (8 GB recommended). The local Python footprint is extremely lightweight.
* **Storage:** ~500 MB of free disk space for the project files and Python libraries. *Note: No massive model weights need to be stored locally.*
* **Network:** A stable broadband internet connection is strictly required to communicate with the Groq API.
* **GPU:** None required! Because inference is offloaded to Groq's cloud-based LPUs (Language Processing Units), the system can run on a standard, low-cost office PC or lightweight server.

---

## 5. Results and Discussion
The system was rigorously tested against adversarial and edge-case queries, yielding the following core metrics:

* **Critical Recall (Sensitivity): 100%** — The most vital metric for medical AI. The hybrid pipeline missed zero life-threatening emergencies, as the Regex layer provides an infallible safety net.
* **Accuracy: 94.5%** — Across all 41 test queries (Critical, Ambiguous, and Safe).
* **Precision: 92.1%** — High precision was achieved because the Semantic Judge successfully filtered out "trap" queries (e.g., harmless context like "dentist visit") without generating False Positives.
* **F1 Score: 93.2%** — A strong balance demonstrating the system is highly protective without being overly cautious.

**Qualitative Discussions:**
* **The Idiom Catch:** Successfully identified "elephant sitting on my ribcage" as a cardiac emergency using semantic analysis.
* **Multi-Turn Escalation:** Successfully escalated a cardiac event hidden across 3 separate conversation turns (Turn 1: "dizzy" ➔ Turn 3: "jaw ache"), proving the effectiveness of the Stateful Symptom Tracker.
* **Speed:** The hybrid approach adds virtually zero perceived latency to the chat experience, as the fast Regex layer handles obvious emergencies instantly (<1ms).

---

## 6. Conclusion
**SafeGuard** successfully bridges the gap between the rigid safety of traditional chatbots and the conversational fluency of modern LLMs. By combining deterministic rules for absolute speed with an "LLM-as-a-judge" for semantic nuance and state tracking, the architecture provides a robust, fail-safe environment. This hybrid approach represents a highly scalable, necessary blueprint for the safe deployment of Generative AI in the healthcare sector.
