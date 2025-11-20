from llm_api import call_llm
from collections import Counter

def simple_agent(question):
    prompt = f"Think step by step and answer this quetion: \n\n{question}"
    answer = call_llm(prompt)
    return answer


def decompose(question):
    prompt = f"""
    Break the question into the *minimum necessary* numbered steps to solve it.

    Rules:
    - 3–5 steps max.
    - Keep each step short (5–15 words).
    - No long explanations.
    - Only return numbered steps.

    QUESTION:
    {question}
"""

    response = safe_call(prompt)
    if response.startswith("Error:"):
        return ["Unable to decompose question."]
    steps = [line.strip() for line in response.split("\n") if line.strip()]

    final_steps = []
    for s in steps:
        if s[0].isdigit():
            final_steps.append(s)
    return final_steps


def solve_step(step):
    prompt = (
        "Solve the following step. Think step by step, but end with 'ANSWER:' and the final result.\n\n"
        f"{step}"
    )

    ans = safe_call(prompt)
    if ans.startswith("Error:"):
        return "Unable to solve step."

    return ans(prompt)


def aggregate(question,step_solutions):
    joined = "\n".join(step_solutions)
    prompt = (
        f"Original question: {question}\n\n"
        f"Here are the solutions to each step:\n{joined}\n\n"
        "Based on everything above, give the final answer only. Format strictly as :" \
        "FINAL: <answer>"
    )

    ans = safe_call(prompt)
    if ans.startswith("Error:"):
        return "Unable to aggregate solutions."

    return ans


def batched_full_agent(question):
    steps = decompose(question)
    step_answers = solve_all_steps_batched(steps)
    

    final = aggregate(question, step_answers)
    return final


def sample_full_agent(question,temperature = 0.7):
    steps = decompose(question)
    step_answers = []
    for step in steps:
        step_answers.append(solve_step(step))

    final = aggregate(question, step_answers)
    return final.strip()

def self_consistent_agent(question,samples = 3,agent_fn = batched_full_agent):
    answers = []
    for _ in range(samples):
        ans = agent_fn(question)
        answers.append(ans)

    cleaned = []
    for a in answers:
        if "FINAL:" in a:
            cleaned.append(a.split("FINAL:")[1].strip())
        else:
            cleaned.append(a.strip())
    
    freq = Counter(cleaned)
    best_answer = freq.most_common(1)[0][0]
    return best_answer

def reflect(question,answer):
    prompt = f"""
You are checking your own work

Question: 
{question}

Proposed answer:
{answer}

Check if the answer is consistent with the question.
If incorrect, provide a corrected final answer

Return format:
VERIFY: correct/incorrect
FINAL: <best-answer>
"""
    a = safe_call(prompt)
    if a.startswith("Error:"):
        return "Unable to reflect on answer."
    return a


def reflective_agent(question,samples = 2):
    base = self_consistent_agent(question,samples=2,agent_fn = batched_full_agent)
    reflection = reflect(question,base)

    if "FINAL:" in reflection:
        return reflection.split("FINAL:")[1].strip()
    return base

def solve_all_steps_batched(steps):
    steps_text = "\n".join(steps)
    prompt = f"""
Solve EACH numbered step below.

For each step, follow the format:
STEP <n> ANSWER: <result>

STEPS:
{steps_text}
"""
    full = safe_call(prompt)
    if full.startswith("Error:"):
        return ["Unable to solve steps."]
    lines = [l for l in full.split("\n") if l.strip()]

    results = []
    for l in lines:
        if "ANSWER:" in l:
            results.append(l.split("ANSWER:")[1].strip())
    return results


def safe_call(prompt,temperature = 0):
    try:
        return call_llm(prompt,temperature=temperature)
    except Exception as e:
        return f"Error: {str(e)}"
    