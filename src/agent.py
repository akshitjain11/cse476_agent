from llm_api import call_llm
from collections import Counter

def simple_agent(question):
    prompt = f"Think step by step and answer this quetion: \n\n{question}"
    answer = call_llm(prompt)
    return answer


def decompose(question):
    prompt = (
        "Break the followuing question into small numbered steps needed to solve it: \n\n"
        f"{question}"
    )

    steps = call_llm(prompt)
    return [line.strip() for line in steps.split("\n") if line.strip()]

def solve_step(step):
    prompt = (
        "Solve the following step. Think step by step, but end with 'ANSWER:' and the final result.\n\n"
        f"{step}"
    )

    return call_llm(prompt)


def aggregate(question,step_solutions):
    joined = "\n".join(step_solutions)
    prompt = (
        f"Original question: {question}\n\n"
        f"Here are the solutions to each step:\n{joined}\n\n"
        "Based on everything above, give the final answer only. Format strictly as :" \
        "FINAL: <answer>"
    )

    return call_llm(prompt)


def full_agent(question):
    steps = decompose(question)
    step_answers = []
    for step in steps:
        step_answers.append(solve_step(step))

    final = aggregate(question, step_answers)
    return final


def sample_full_agent(question,temperature = 0.7):
    steps = decompose(question)
    step_answers = []
    for step in steps:
        step_answers.append(solve_step(step))

    final = aggregate(question, step_answers)
    return final.strip()

def self_consistent_agent(question,samples = 3):
    answers = []
    for _ in range(samples):
        ans = sample_full_agent(question,temperature=0.7)
        answers.append(ans)

    cleaned = []
    for a in answers:
        if "FINAL:" in a:
            cleaned.append(a.split("FINAL:")[1].strip())
        else:
            cleaned.append(a.strip())
    
    freq = Counter(cleaned)
    best_answer, _ = freq.most_common(1)[0]

    return best_answer

def reflect(question,answer):
    prompt = f"""
You are checking your own work.frozenset

Question: 
{question}

Proposed answer:
{answer}

Check if the answer is consistent with the question.
If incorrect, provide a corrected final answer.frozenset

Return format:
VERIFY: correct/incorrect
FINAL: <best-answer>
"""
    return call_llm(prompt)


def reflective_agent(question,samples = 2):
    base = self_consistent_agent(question,samples=2)
    reflection = reflect(question,base)

    if "FINAL:" in reflection:
        return reflection.split("FINAL:")[1].strip()
    return base

def solve_all_steps_batched(steps):
    steps_text = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(steps))
    prompt = f"""
Solve each of the following steps. Think step by step, but for each step end with:
STEP <n> ANSWER: <answer>

STEPS:
{steps_text}
"""
    full = call_llm(prompt)
    return full.split("\n")
    