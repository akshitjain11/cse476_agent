# cse476_agent — LLM-based Judgment Detector / Agent

This repository contains a small LLM-driven agent and utilities used for the CSE476 final project. The agent implements decomposition, step solving, aggregation, and simple reflective/self-consistent strategies for producing answers.

## Project structure


- `cse476_final_project_dev_data.json` — developer dataset (example input/outputs)
- `final_project_tutorial.ipynb` — notebook with experiments and examples
- `src/` — source code
	- `agent.py` — primary agent implementation (decompose/solve/aggregate/reflect)
	- `run.py` — small runner demonstrating `reflective_agent`
	- `generate_answer_template.py` — generates an answers JSON for autograder; reads `cse_476_final_project_test_data.json` by default
    - `llm_api.py` — simple wrapper around the LLM HTTP API

## Requirements

The project uses Python 3.10+ (tested on 3.11+). The main runtime dependencies are:

- requests
- pandas
- numpy
- scikit-learn
- matplotlib



## Quick start — run the reflective agent

From the `src` folder run:

```powershell
cd src
python run.py
```

This runs the `reflective_agent(...)` example in `run.py` and prints the answer and the total number of LLM calls.

## Generate answers for the autograder

The helper `generate_answer_template.py` reads test questions from `../cse_476_final_project_test_data.json` and writes answers to `../cse_476_final_project_answers.json`.

Run it from `src`:

```powershell
python generate_answer_template.py
```


