# main.py
"""
Interactive Safeguard LLM Testing
"""

from core.pipeline import safe_guard
from core.evaluation import run_evaluation
from core.session_state import SessionState, reset_session
from core.risk import reset_conversation_state  # <-- ADD THIS LINE
import sys


def interactive_mode():
    """Interactive testing mode"""
    print("="*60)
    print("🏥 Medical Safeguard LLM - Interactive Mode")
    print("="*60)
    print("Commands: 'exit', 'eval', 'redteam', 'help', 'clear'")
    print("-"*60)
    
    conversation_history = []
    session_state = SessionState()
    
    while True:
        query = input("\nYou: ").strip()
        
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        if query.lower() == "help":
            print("""
Available commands:
  exit     - Exit the program
  eval     - Run evaluation suite
  redteam  - Run adversarial red-team testing
  help     - Show this help
  clear    - Clear conversation history
            """)
            continue
        
        if query.lower() == "clear":
            conversation_history = []
            reset_conversation_state()  # <-- ADD THIS LINE
            print("Conversation history and risk memory cleared.")
            continue
        
        if query.lower() == "eval":
            run_evaluation()
            continue

        if query.lower() == "redteam":
            try:
                from red_team import run_red_team
                run_red_team()
            except ImportError:
                print("Failed to import red_team module. Is it created?")
            continue
        
        # Process through safeguard
        result = safe_guard(query, conversation_history, session_state=session_state)
        
        # Add to history
        conversation_history.append(query)
        conversation_history.append(result.response)
        
        # Display result
        print("\n" + "-"*40)
        if result.rule_triggered:
            print("🚨 EMERGENCY RULE TRIGGERED")
        elif result.response_modified:
            print("⚠️ RESPONSE SANITIZED")
        else:
            print("✅ LLM RESPONSE")
        print("-"*40)
        
        print(f"\n{result.response}")
        
        # Debug info (optional)
        print(f"\n📊 [Label: {result.label} | Risk: {result.risk_score:.2f} | "
              f"Time: {result.response_time_ms:.0f}ms]")
        if session_state.detected_symptoms:
            print(f"🏥 Symptoms: {', '.join(session_state.detected_symptoms)}")
        if getattr(result, 'judge_score', None) is not None:
            print(f"⚖️ Judge Score: {result.judge_score}/10 (Safe: {result.judge_safe})")


def evaluation_mode():
    """Run full evaluation"""
    run_evaluation()


def single_test(query: str):
    """Test single query"""
    result = safe_guard(query)
    
    print(f"\nQuery: {query}")
    print(f"Label: {result.label}")
    print(f"Risk: {result.risk_score:.2f} ({result.risk_category})")
    print(f"Rule: {result.rule_triggered}")
    print(f"Response: {result.response}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "eval":
            evaluation_mode()
        elif sys.argv[1] == "redteam":
            try:
                from red_team import run_red_team
                run_red_team()
            except ImportError:
                print("Failed to import red_team module.")
        else:
            single_test(" ".join(sys.argv[1:]))
    else:
        interactive_mode()