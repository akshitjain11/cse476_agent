from agent import (reflective_agent)
from llm_api import CALL_COUNT

if __name__ == "__main__":
    q = "What is the product of the real roots of the equation $x^2 + 18x + 30 = 2 \sqrt{x^2 + 18x + 45}$??"

    print("\n=== REFLECTIVE AGENT ===")
    print(reflective_agent(q))
    print(f"Total LLM calls: {CALL_COUNT}")
