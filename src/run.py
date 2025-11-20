from agent import (simple_agent, decompose, solve_step, aggregate, full_agent, sample_full_agent, self_consistent_agent,reflective_agent)
from llm_api import CALL_COUNT

if __name__ == "__main__":
    q = "A gardener plants three maple trees, four oaks, and five birch trees in a row. He plants them in random order, each arrangement being equally likely. Let $\\frac m n$ in lowest terms be the probability that no two birch trees are next to one another. Find $m+n$ ."

    print("\n=== REFLECTIVE AGENT ===")
    print(reflective_agent(q, samples=2))
    print(f"Total LLM calls: {CALL_COUNT}")
