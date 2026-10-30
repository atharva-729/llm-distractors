"""Our own example questions (not Eedi data) run through every model with the frozen
prompts, so the public page can show real model answers without publishing Eedi questions.

Usage: python src/site_examples.py  ->  results/site_examples.jsonl
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402
from llm import LLM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "claude-haiku-4-5", "gemini-2.5-flash-lite", "gpt-5-mini",
          "gpt-5.4", "gpt-5.4-think", "claude-sonnet-4-6", "gemini-3.1-pro"]

# Written for this page. Options are listed in display order; each wrong option comes
# with the misconception that produces it, and an unrelated misconception for the control.
QUESTIONS = [
    {"id": "fractions", "text": "Work out 1/2 + 1/3", "correct": "B",
     "options": {"A": "2/5", "B": "5/6", "C": "1/6", "D": "2/6"},
     "misconceptions": {"A": "Adds the numerators together and adds the denominators together",
                        "C": "Multiplies the fractions when asked to add them",
                        "D": "Adds the numerators but multiplies the denominators"},
     "control": "Thinks that the angles in a triangle add up to 360 degrees"},
    {"id": "order", "text": "Work out 3 + 4 × 2", "correct": "C",
     "options": {"A": "14", "B": "9", "C": "11", "D": "24"},
     "misconceptions": {"A": "Always works through a calculation from left to right, one step at a time",
                        "B": "Adds every number in the calculation instead of following the operations",
                        "D": "Multiplies every number in the calculation instead of following the operations"},
     "control": "Believes a percentage of a number is found by dividing the number by the percentage"},
]


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    llm = LLM(cfg)
    rows = []
    for q in QUESTIONS:
        shown = [q["options"][L] for L in prompts.LETTERS]
        jobs = [("none", None, None)] + [(c, L, q["misconceptions"][L] if c == "matched" else q["control"])
                                          for L in q["misconceptions"] for c in ("matched", "mismatched")]
        for model in MODELS:
            for cond, target, persona in jobs:
                framing = "none" if cond == "none" else "student"
                msgs = prompts.single_messages(framing, persona, q["text"], shown)
                r = llm.chat(model, msgs, 20, prompts.ANSWER_SCHEMA)
                pick = prompts.parse_single(r["text"])
                rows.append({"question": q["id"], "model": model, "condition": cond, "target": target,
                             "pick": pick, "raw": r["text"]})
    out = ROOT / cfg["paths"]["results_dir"] / "site_examples.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} answers, new calls {llm.session['calls']}, ${llm.session['usd']:.4f} -> {out}")


if __name__ == "__main__":
    main()
