"""Phase 1 metrics from results/{stage}_<model>_responses.jsonl (all models).

Usage: python src/analyze.py --stage pilot [--show 10]

Writes results/{stage}_metrics.csv. --show prints random single-mode responses
with question text to the console only (question text is not saved; see LICENSE_NOTES.md).
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
GROUP = ["model", "mode", "framing"]


def rates(df, outcomes):
    n = len(df)
    return {f"{o}_rate": (df.outcome == o).sum() / n if n else float("nan") for o in outcomes} | {"n": n}


def metrics(df):
    rows = []
    personas = df[df.condition != "none"]
    for key, g in personas.groupby(GROUP):
        m = g[g.condition == "matched"]
        mm = g[g.condition == "mismatched"]
        r = dict(zip(GROUP, key))
        mr = rates(m, ["hit", "slip", "wrong_stupid", "invalid"])
        r.update({f"matched_{k}": v for k, v in mr.items()})
        r["mismatched_hit_rate"] = (mm.outcome == "hit").mean()
        r["mismatched_slip_rate"] = (mm.outcome == "slip").mean()
        r["mismatched_invalid_rate"] = (mm.outcome == "invalid").mean()
        r["lift"] = r["matched_hit_rate"] - r["mismatched_hit_rate"]
        rows.append(r)
    out = pd.DataFrame(rows)
    out = out.merge(solvable_lift(df), on=GROUP, how="left")
    nop = df[df.condition == "none"].groupby(["model", "mode"]).outcome.agg(
        no_persona_accuracy=lambda s: (s == "correct").mean(),
        no_persona_invalid=lambda s: (s == "invalid").mean())
    return out.merge(nop.reset_index(), on=["model", "mode"], how="left")


def solvable_lift(df):
    """Lift restricted to questions the same model/mode answers correctly with no persona.
    A hit on a question the model can't solve may be an accident, not play-acting."""
    solved = df[(df.condition == "none") & (df.outcome == "correct")][["model", "mode", "question_id"]]
    s = df[df.condition != "none"].merge(solved, on=["model", "mode", "question_id"])
    rows = []
    for key, g in s.groupby(GROUP):
        hit = lambda c: (g[g.condition == c].outcome == "hit").mean()
        rows.append(dict(zip(GROUP, key)) | {
            "solvable_n": (g.condition == "matched").sum(),
            "solvable_matched_hit_rate": hit("matched"),
            "solvable_mismatched_hit_rate": hit("mismatched"),
            "solvable_lift": hit("matched") - hit("mismatched")})
    return pd.DataFrame(rows, columns=GROUP + ["solvable_n", "solvable_matched_hit_rate",
                                               "solvable_mismatched_hit_rate", "solvable_lift"])


def agreement(df):
    """Same item answered in single vs batched mode: how often the same option?"""
    if df["mode"].nunique() < 2:
        return None
    key = ["model", "question_id", "target_letter", "condition", "framing"]
    w = df.pivot_table(index=key, columns="mode", values="picked_letter", aggfunc="first", dropna=False)
    w = w.dropna()
    w["agree"] = w["single"] == w["batched"]
    return w.reset_index().groupby(["condition", "framing"]).agree.agg(["mean", "count"])


def position_bias(df):
    return (df[df.condition != "none"].groupby(["mode", "framing"]).shown_letter
            .value_counts(normalize=True).unstack().round(2))


def show(df, pairs, n, seed):
    qs = {p["question_id"]: p for p in pairs}
    names = {p["misconception_id"]: p["misconception_name"] for p in pairs}
    single = df[(df["mode"] == "single") & (df.condition != "none")]
    for _, r in single.sample(n, random_state=seed).iterrows():
        q = qs[r.question_id]
        print("\n" + "-" * 70)
        print(f"Q{r.question_id}  {r.framing} / {r.condition}  ->  {r.outcome.upper()}")
        print(f"persona: {names.get(r.persona_misconception_id)}")
        if r.condition == "mismatched":
            print(f"(tagged on target: {names.get(r.tagged_misconception_id)})")
        print(q["question_text"])
        for L, t in q["options"].items():
            tags = []
            if L == r.correct_letter: tags.append("correct")
            if L == r.target_letter: tags.append("target distractor")
            if L == r.picked_letter: tags.append("<-- PICKED")
            print(f"  {L}) {t}  {' '.join(tags)}")
        print(f"raw reply: {r.raw!r}  (shown as {r.shown_letter}, display order {r.perm})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    res = ROOT / cfg["paths"]["results_dir"]
    files = sorted(res.glob(f"{args.stage}_*_responses.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    print(f"read {len(files)} files: {[f.name for f in files]}")

    m = metrics(df)
    m.to_csv(res / f"{args.stage}_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    cols = GROUP + ["matched_n", "matched_hit_rate", "mismatched_hit_rate", "lift", "matched_slip_rate",
                    "matched_wrong_stupid_rate", "matched_invalid_rate", "no_persona_accuracy",
                    "solvable_n", "solvable_lift"]
    print("=== Metrics ===")
    print(m[cols].round(3).to_string(index=False))
    print("\n=== Parse / retry ===")
    print(df.groupby("mode").agg(rows=("outcome", "size"), invalid=("outcome", lambda s: (s == "invalid").sum()),
                                 retried=("retried", "sum")).to_string())
    a = agreement(df)
    if a is not None:
        print("\n=== Single vs batched: same option picked ===")
        print(a.round(3).to_string())
    print("\n=== Shown-letter distribution (position bias check) ===")
    print(position_bias(df).to_string())
    print(f"\nwrote {res / f'{args.stage}_metrics.csv'}")

    if args.show:
        pairs = [json.loads(l) for l in open(ROOT / cfg["paths"]["processed_dir"] / "pairs.jsonl", encoding="utf-8")]
        show(df, pairs, args.show, cfg["seed"])


if __name__ == "__main__":
    main()
