"""
Evals for the Flipp agent.

Runs each test question several times, records which tools the agent called,
checks the results with code checks + an LLM judge, and prints a scorecard.

Usage (from the same folder as flipp.py):
    python evals.py
"""

import contextlib
import io
import json

import flipp  # your agent file - nothing in it needs to change

RUNS_PER_QUESTION = 3
JUDGE_MODEL = "claude-sonnet-5-5"   # small, cheap model is fine for grading

# ---------- 1. Record every tool call without touching the agent ----------
# The agent looks tools up in flipp.TOOL_FUNCTIONS, so we wrap each one
# with a recorder. Same tools, but now we can see every input and output.

calls = []

def make_recorder(name, fn):
    def wrapper(**kwargs):
        try:
            output = fn(**kwargs)
        except Exception as e:
            output = {"exception": str(e)}
            calls.append({"tool": name, "input": kwargs, "output": output})
            raise
        calls.append({"tool": name, "input": kwargs, "output": output})
        return output
    return wrapper

print(f"old tool function {flipp.TOOL_FUNCTIONS}");

for tool_name, tool_fn in list(flipp.TOOL_FUNCTIONS.items()):
    flipp.TOOL_FUNCTIONS[tool_name] = make_recorder(tool_name, tool_fn)
print(f"updated tool function {flipp.TOOL_FUNCTIONS}");

# ---------- 2. Graders ----------
# Every check takes (question, answer, calls) and returns True/False.

def used(tool):
    def check(q, a, c):
        passed = any(x["tool"] == tool for x in c)
        reason = f"{tool} was used." if passed else f"{tool} was not used."
        return reason, passed
    return check


def no_tools(q, a, c):
    test = len(c) == 0
    if test:
        return f"No tool is used.", test
    return f"Tools are/is used." , test

def searched_postal(code):
    def check(q, a, c):
        passed = any(
            x["tool"] == "search_flipp"
            and code in x["input"].get("postal_code", "").replace(" ", "").upper()
            for x in c
        )
        reason = f"Postal code {code} was used." if passed else f"Postal code {code} was not used."
        return reason, passed
    return check

def judge(rubric):
    """LLM-as-judge: a second model grades the answer against the tool results."""
    def check(q, a, c):
        prompt = (
            "You are grading an AI grocery-deal assistant.\n\n"
            f"The assistant's instructions (it is allowed to follow these):\n{flipp.SYSTEM}\n\n"
            f"User question: {q}\n\n"
            f"Tool calls and results (this is the ground truth):\n{json.dumps(c, default=str)[:8000]}\n\n"
            f"Assistant's final answer:\n{a}\n\n"
            f"Rubric: {rubric}\n\n"
            "First explain your reasoning in 2-3 sentences, then give PASS or FAIL alone on the last line."
        )
        r = flipp.client.messages.create(
            model=JUDGE_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = "".join(b.text for b in r.content if b.type == "text").strip()
        if not verdict:
            return f"Judge gave no verdict (stop_reason: {r.stop_reason})", False
        last_line = verdict.splitlines()[-1].strip("* ").upper()
        return verdict, last_line.startswith("PASS")
    return check

GROUNDED = judge(
    "Every product, store and price in the answer must appear in the tool results. "
    "Anything invented or not supported by the tool results is a FAIL."
    "Judge only whether facts match the tool results, not how the answer was produced."
)
PICKS_LOWEST = judge(
    "If the tool results include sizes, the answer must recommend the option with the lowest "
    "price_per_unit value. If sizes are missing, the answer must say it can't compare value "
    "and must not guess sizes."
)
FLAGS_AMBIGUITY = judge(
    "An item counts as unclear ONLY if it has no size, or its price is a multi-buy "
    "(e.g. '2/$7', '2 for 5'). Plain numbers like '4' or '5.98' are normal prices. "
    "The answer must mention each unclear item in any wording (e.g. 'no size given', "
    "'couldn't compare'). If no items are unclear, PASS."
)


# ---------- 3. The test set ----------

TEST_CASES = [
    {"question": "what's the best value milk this week?",
     "checks": {"searched": used("search_flipp"), "calculated": used("price_per_unit"),
                "picks lowest": PICKS_LOWEST, "grounded": GROUNDED}},
    {"question": "what's the best value strawberries this week?",
     "checks": {"searched": used("search_flipp"), "flags ambiguity": FLAGS_AMBIGUITY,
                "grounded": GROUNDED}},
    {"question": "cheapest eggs near M5V 1J1",
     "checks": {"used postal code": searched_postal("M5V1J1"), "grounded": GROUNDED}},
    {"question": "what is the capital of France?",
     "checks": {"no tools": no_tools}},
    {"question": "price of bananas this week",
     "checks": {"searched": used("search_flipp"), "grounded": GROUNDED}},
]


# ---------- 4. Run everything and print the scorecard ----------

def run_once(question):
    calls.clear()
    with contextlib.redirect_stdout(io.StringIO()):   # hide the agent's debug prints
        answer = flipp.run_agent(question)
    return answer, list(calls)

failures = []
print(f"Running {len(TEST_CASES)} questions x {RUNS_PER_QUESTION} runs...\n")

for case in TEST_CASES:
    q = case["question"]
    scores = {name: 0 for name in case["checks"]}

    for run in range(RUNS_PER_QUESTION):
        try:
            answer, run_calls = run_once(q)
        except Exception as e:
            failures.append({"question": q, "run": run + 1, "error": str(e)})
            continue
        for name, check in case["checks"].items():
            reason, test = check(q, answer, run_calls)
            if test:
                scores[name] += 1
            else:
                failures.append({"question": q, "run": run + 1, "failed_check": name, "reason":reason,
                                 "answer": answer, "tool_calls": run_calls})

    print(q)
    for name, passed in scores.items():
        mark = "OK " if passed == RUNS_PER_QUESTION else "!! "
        print(f"   {mark}{name:<18} {passed}/{RUNS_PER_QUESTION}")
    print()

with open("eval_failures.json", "w") as f:
    json.dump(failures, f, indent=2, default=str)
print(f"{len(failures)} failed checks saved to eval_failures.json")