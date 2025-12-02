import json
from agent import reflective_agent
from llm_api import CALL_COUNT

with open("../cse476_final_project_dev_data.json") as f:
    data = json.load(f)

outputs = []
for item in data:
    q = item["question"]
    CALL_COUNT = 0

    ans = reflective_agent(q)

    outputs.append({
        "id": item["id"],
        "question": q,
        "answer": ans,
        "calls": CALL_COUNT
    })

with open("./cse_476_final_project.dev_outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)