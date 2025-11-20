from agent import (simple_agent, decompose, solve_step, aggregate, full_agent, sample_full_agent, self_consistent_agent,reflective_agent)
from llm_api import CALL_COUNT

if __name__ == "__main__":
    q = "How many even integers between 4000 and 7000 have four different digits?"

    print("\n=== REFLECTIVE AGENT ===")
    print(reflective_agent(q, samples=2))
    print(f"Total LLM calls: {CALL_COUNT}")
