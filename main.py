# main.py
"""
Interactive Safeguard LLM Testing
"""

from core.pipeline import safe_guard
from core.evaluation import SafeguardEvaluator, create_test_dataset, run_evaluation
import sys


def interactive_mode():
    """Interactive testing mode"""
    print("="*60)
    print("🏥 Medical Safeguard LLM - Interactive Mode")
    print("="*60)
    print("Commands: 'exit', 'eval', 'help'")
    print("-"*60)
    
    conversation_history = []
    
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
  help     - Show this help
  clear    - Clear conversation history
            """)
            continue
        
        if query.lower() == "clear":
            conversation_history = []
            print("Conversation history cleared.")
            continue
        
        if query.lower() == "eval":
            run_evaluation()
            continue
        
        # Process through safeguard
        result = safe_guard(query, conversation_history)
        
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
        else:
            single_test(" ".join(sys.argv[1:]))
    else:
        interactive_mode()