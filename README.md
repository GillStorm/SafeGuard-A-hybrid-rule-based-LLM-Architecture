# Medical Safeguard LLM Pipeline

A hybrid rule-based and LLM architecture designed to ensure safe, reliable, and medically appropriate responses to health-related queries. This system acts as a protective layer over generative models (like Llama 3.1) to prevent the generation of medical diagnoses, inappropriate advice, or unsafe recommendations.

## Features

- **Query Classification**: Categorizes user inputs into Critical, Ambiguous, Non-Critical, or Off-Topic domains.
- **Risk Assessment Engine**: Evaluates the severity and context of the query to determine if the LLM should be allowed to answer or if a predefined safe fallback should be used.
- **Emergency Rule Trigger**: Immediately identifies critical keywords (e.g., "chest pain", "heart attack") and forcefully overrides the LLM to recommend seeking immediate emergency medical attention.
- **Response Validation (Sanitization)**: Post-processes the LLM's output to ensure disclaimers are present and unsafe advice is filtered out.
- **Interactive Mode**: Provides a CLI interface to chat with the safeguard system in real-time.
- **Evaluation Suite**: Built-in evaluation dataset to measure system performance and adherence to safety guidelines.

## Architecture

The pipeline consists of the following components:

1. **Classifier (`core/classifier.py`)**: Determines the domain and severity of the input query based on keyword analysis.
2. **Risk Analyzer (`core/risk.py`)**: Computes a holistic risk score based on the classification and conversational context.
3. **Rules Engine (`core/rules.py`)**: Applies deterministic safety rules and emergency overrides.
4. **LLM Interface (`core/llm.py`)**: Connects to the Groq API (Llama 3.1) with a strict medical safety system prompt.
5. **Main Pipeline (`core/pipeline.py`)**: Orchestrates the classification, risk assessment, rule checking, and LLM generation.

## Getting Started

### Prerequisites

- Python 3.8+
- `openai` python package

### Setup

1. Install dependencies:
   ```bash
   pip install openai pandas scikit-learn
   ```

2. Set your Groq API key:
   ```powershell
   $env:GROQ_API_KEY="your_api_key_here"
   ```

### Usage

Run the interactive CLI:
```bash
python main.py
```

Run the evaluation suite:
```bash
python main.py eval
```

Run a single query test:
```bash
python main.py "I have a headache"
```

## Disclaimer

This software is for educational and research purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
