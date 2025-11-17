from llm_api import call_llm

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
    return steps.split("\n")

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
        "Based on everything above, give the final answer only. Formate: Answer: <final>"
    )

    return call_llm(prompt)


def full_agent(question):
    steps = decompose(question)
    step_answers = []
    for step in steps:
        step_answers.append(solve_step(step))

    final = aggregate(question, step_answers)
    return final