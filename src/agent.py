from llm_api import call_llm
from collections import Counter
import re

def simple_agent(question):
    prompt = f"""Think step by step and answer this question concisely.

RULES:
- Provide a direct, clear answer
-No markdown formatting unless specified
-No bold text or special formatting
-Just give the answer

QUESTION:
{question}

Answer:"""

    answer = call_llm(prompt)
    return answer.strip()


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

    return ans

def is_multiple_choice(question:str) -> bool:
    return ("A." in question or "A)" in question or "(A)" in question or "Options:" in question or "options:" in question)


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

def is_math_question(question:str) -> bool:
    keywords = [
        "midpoint", "quadratic", "equation", "solve for x", "integral",
        "derivative", "function", "calculate", "what is", "value of","equations",
        "algebra", "geometry", "trigonometry", "logarithm", "probability","parentheses",
        "divide", "multiply", "add", "subtract", "sum", "product","roots","integer","integers","perfect square","power of",
        "frac"
    ]

    q = question.lower()
    if any(s in q for s in ["$","=","+","*","^"]):
        return True
    return any(kw.lower() in q for kw in keywords)

def is_planning_task(question:str) -> bool:
    keywords = [
        "[PLAN]",
        "[STATEMENT]",
        "actions i can do",
        "restrictions on my actions",
        "initial conditions",
        "my goal is to have"
    ]
    q = question.lower()
    return any(kw.lower() in q for kw in keywords)

def planning_agent(question:str) -> str:
    prompt = f"""
    You are an expert at sequential planning and reasoning

CRITICAL RULES:
- Analyze the PROBLEM very carefully, paying attention to ALL constraints
- Think through each action step by step
-Verify that each action's preconditions are satisfied before proceeding
- Track the state changes after each action
-Make sure the final state matches the GOAL exactly
-Output ONLY the [PLAN] section with valid actions
-Do NOT include explanations or reasoning

PROBLEM:
{question}

YOUR PLAN (only [PLAN] ... [PLAN END]):
"""
    plan = safe_call(prompt,temperature=0.2)
    if plan.startswith("Error:"):
        return "Unable to generate plan."
    
    if "[PLAN]" in plan:
        start = plan.find("[PLAN]")
        end = plan.find("[PLAN END]")
        if end>start:
            return plan[start:end+10].strip()
        else:
            return plan[start:].strip()
    return plan.strip()

def math_agent(question:str) -> str:
    prompt = f"""
    You are an expert AIME competition solver.

Rules:
-Answers are ALWAYS integers between 0 and 999.
-Show your mathematical reasoning clearly
-Perform exact calculations
-Double check arithmetic
-State the final answer clearly at the end as "The answer is: X"

QUESTION:
{question}
"""
    ans = safe_call(prompt,temperature=0)
    if ans.startswith("Error:"):
        return "Unable to solve math question."
    
    patterns = [
        r"The answer is[:\s]+(-?\d+)",
        r"Final answer[:\s]+(-?\d+)",
        r"ANSWER[:\s]+(-?\d+)",
        r"=-?\d+",
    ]
    for pattern in patterns:
        match = re.search(pattern,ans.lower())
        if match:
            num = int(match.group(1))
            if 0<=num<=999:
                return str(num)
    nums = re.findall(r"-?\d+\.?\d*", ans)
    if nums:
        return nums[-1].strip()
    return ans.strip()

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

    if is_planning_task(question):
        plan = planning_agent(question)
        return plan

    if is_multiple_choice(question):
        prompt = f"Select the correct option. Answer ONLY with the letter. (A, B, C, or D)\n\nQuestion:\n{question}"
        out=safe_call(prompt)
        if out.startswith("Error:"):
            return "Unable to answer multiple choice question."
        out_clean = out.strip()
        for char in out_clean:
            if char.upper() in "ABCDEFGH":
                return char.upper()
        return out_clean

    if is_coding_task(question):
        code = coding_agent(question)
        return code
    
    if is_math_question(question):
        math_ans = math_agent(question)
        return math_ans
    
    simple = [
        "facts:", "context:", "which", "what", "does", "is", "are",
        "would", "could", "should"
    ]
    q_lower = question.lower()
    is_simple = any(ind in q_lower for ind in simple) and len(question)<1000

    if is_simple:
        return simple_agent(question)
    
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


def safe_call(prompt,temperature = 0,max_retries=2):
    for attempt in range(max_retries):
        try:
            result =  call_llm(prompt,temperature=temperature)
            if result and len(result.strip())>0:
                return result
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"
            continue
    return "Error: Maximum retries exceeded."
    

def is_coding_task(question:str) -> bool:
    coding_keywords = [
        "you should write self-contained code",
        "you should write code",
        "write a function",
        "implement a function",
        "create a function",
        "def ",
        "python"
        "write python code",
        "starting with:",
        "def task_func(",
        "import ",
        "sklearn",
        "pandas",
        "matplotlib",
        "numpy",
        "requests",
        "sqlite",
        "random_seed",
    ]
    question_lower = question.lower()
    if "def " in question_lower or "write" in question_lower and "function" in question_lower:
        return True
    return any(keyword in question_lower for keyword in coding_keywords)

def coding_agent(question:str) -> str:
    prompt = f"""
    You are an expert Python programmer.

    Critical Rules:
    - Write ONLY the function code, nothing else
    - Include necessary imports at the top
    - Use proper Python syntax
    - No explanations, no markdown, no test cases
    - Return working code only

QUESTION:
{question}
"""
    code = safe_call(prompt)
    if code.startswith("Error:"):
        return "Unable to generate code."
    return code
    

def extract_final_answer(text):
    if not text:
        return "No answer found."
    if "FINAL:" in text:
        return text.split("FINAL:")[1].strip()
    
    return text.strip()
    