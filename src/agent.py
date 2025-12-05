from llm_api import call_llm
from collections import Counter
import re

def log_agent_use(agent_name: str, question: str):
    print(f"ROUTING QUESTION TO AGENT: {agent_name}")

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
    if len(question) > 250:   
        return False
    q = question.lower()
    if any(x in q for x in ["context:", "according to", "based on the following", "passage"]):
        return False

    math_patterns = [
         "how many", "difference", "sum", "total", "left", "gave",
        "increase", "decrease", "less than", "more than",
        "ratio", "average", "mean", "probability"
    ]
    if any (p in q for p in math_patterns):
        return True
    keywords = [
        "midpoint", "quadratic", "equation", "solve for x", "integral",
        "derivative", "function", "calculate", "value of","equations",
        "algebra", "geometry", "trigonometry", "logarithm", "probability","parentheses",
        "divide", "multiply", "add", "subtract", "sum", "product","roots","integer","integers","perfect square","power of",
        "frac","percent","%"
    ]

    
    if "24-game challenge" in q:
        return False
    
    if any(s in q for s in ["$","=","+","*","^"]):
        return True
    return any(kw.lower() in q for kw in keywords)

def is_24_game_question(question:str) -> bool:
    return "24-game challenge" in question.lower()

def game_24_agent(question:str) -> str:
    prompt = f"""
    You are an expert at solving 24-game puzzle.
    

CRITICAL RULES:
- Use each number exactly once
- Only use operations +, -, *, / and parentheses
- Find an expression that equals exactly 24
-Reply ONLY with: Solution: <expression>
-No explanations, no other text

CHALLENGE:
{question}

YOUR ANSWER:
"""
    result = safe_call(prompt,temperature=0.3)
    if result.startswith("Error:"):
        return "Unable to solve 24-game challenge."
    if "Solution:" in result:
        for line in result.split("\n"):
            if "Solution:" in line:
                return line.strip()
    
    return result.strip()

def has_context(question: str) -> bool:
    context_keywords = [
        "context:","facts:","[par]","[doc]","[tle]","given that","according to","based on"
    ]
    q = question.lower()
    return any(kw in q for kw in context_keywords)

def context_agent(question:str) -> str:
    prompt = f"""
    You are an expert at answering questions based on provided context.

CRITICAL RULES:
-Read the CONTEXT carefully
-Find the answer directly from the CONTEXT
-Give a concise, direct answer
-Do not add information not in the CONTEXT
-If the answer is a title/name, give it exactly as shown

QUESTION WITH CONTEXT:
{question}

YOUR ANSWER:
"""
    answer = safe_call(prompt,temperature=0)
    if answer.startswith("Error:"):
        return "Unable to answer question with context."
    return answer.strip()


def is_easy_math_question(question:str) -> bool:
    q = question.lower()
    if re.fullmatch(r"[0-9\+\-\*\/\(\)\s]+", q):
        return True

    easy_keywords = [
        "what is", "calculate", "find", "value of",
        "sum", "difference", "product", "quotient",
        "plus", "minus", "times", "divided by", "how many","how much","what is","calculate","find","pounds","dollars","percent""times as much","more than", "fewer than","difference","sum","total","sold","bought","weigh","weighed","cost","pay","commission"
        ]

    if any(kw in q for kw in easy_keywords):
        num_count = len(re.findall(r"\d+", q))
        if num_count <=6:
            return True

    return False

def easy_math_agent(question: str) -> str:
    prompt = f"""
You are an expert at solving simple arithmetic and short word problems.

RULES:
- Solve the problem in your head.
- Do NOT show steps.
- Do NOT explain.
- Answer ONLY with the final number.
- No words, no punctuation, no labels.

QUESTION:
{question}

FINAL ANSWER:"""

    ans = safe_call(prompt, temperature=0)
    ans = ans.strip()
    ans = ans.replace("FINAL ANSWER:", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", ans)
    if nums:
        return nums[-1]

    return ans

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
        log_agent_use("PLANNING_AGENT", question)
        plan = planning_agent(question)
        return plan
    
    if is_24_game_question(question):
        log_agent_use("24_GAME_AGENT", question)
        solution = game_24_agent(question)
        return solution
    
    if has_context(question):
        log_agent_use("CONTEXT_AGENT", question)
        context_ans = context_agent(question)
        return context_ans

    if is_multiple_choice(question):
        log_agent_use("MULTIPLE_CHOICE_AGENT", question)
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
        log_agent_use("CODING_AGENT", question)
        code = coding_agent(question)
        return code
    
    
    
    if is_math_question(question):
        if is_easy_math_question(question):
            log_agent_use("EASY_MATH_AGENT",question)
            return easy_math_agent(question)
        else:
            log_agent_use("MATH_AGENT", question)
            math_ans = math_agent(question)
            return math_ans
    
    
    simple = [
        "facts:", "context:", "which", "what", "does", "is", "are",
        "would", "could", "should","why","how"
    ]
    q_lower = question.lower()
    is_simple = any(ind in q_lower for ind in simple)

    if is_simple or len(question.split())<60:
        log_agent_use("SIMPLE_AGENT", question)
        return simple_agent(question)
    
    log_agent_use("REFLECTIVE_REASONING_AGENT", question)
    #base = self_consistent_agent(question,samples=2,agent_fn = batched_full_agent)
    #reflection = reflect(question,base)

    #if "FINAL:" in reflection:
     #   return reflection.split("FINAL:")[1].strip()
    #return base
    return batched_full_agent(question)

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
    