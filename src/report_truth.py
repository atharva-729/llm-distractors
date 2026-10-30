"""Second local report: can a model stop knowing the answer? (the pull of the correct answer)

Usage: python src/report_truth.py --stage full  ->  reports/{stage}_truth.html

Reuses Phase 1 results only (no new LLM calls). Embeds Eedi question text for the
example pairs, so like report.py it is for local viewing only (see LICENSE_NOTES.md).
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
CODE = {"hit": "h", "slip": "s", "wrong_stupid": "w", "invalid": "i", "correct": "c", "incorrect": "x"}


def load(cfg, stage):
    files = sorted((ROOT / cfg["paths"]["results_dir"]).glob(f"{stage}_*_responses.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df = df[df["mode"] == "single"]
    nop = (df[df.condition == "none"][["model", "question_id", "picked_letter", "outcome"]]
           .rename(columns={"picked_letter": "np_pick", "outcome": "np_out"}))
    p = df[df.condition != "none"].merge(nop, on=["model", "question_id"])
    p["solved"] = p.np_out == "correct"
    return df, p


def per_model(p):
    """Correct-answer rates with and without a persona, and where the misses go."""
    out = []
    for (model, framing), g in p.groupby(["model", "framing"]):
        m, c = g[g.condition == "matched"], g[g.condition == "mismatched"]
        miss = m[(m.outcome != "hit") & (m.outcome != "invalid")]
        r = {"model": model, "framing": framing, "n": len(m),
             "acc_none": float(m.drop_duplicates("question_id").solved.mean()),
             "correct_control": float((c.outcome == "slip").mean()),
             "correct_matched": float((m.outcome == "slip").mean()),
             "hit_matched": float((m.outcome == "hit").mean()),
             "hit_control": float((c.outcome == "hit").mean()),
             "other_matched": float((m.outcome == "wrong_stupid").mean()),
             "misses": len(miss), "truth_share": float((miss.outcome == "slip").mean())}
        for solved, s in m.groupby("solved"):
            k = "solved" if solved else "unsolved"
            r[f"{k}_n"] = len(s)
            r[f"{k}_hit"] = float((s.outcome == "hit").mean())
            r[f"{k}_slip"] = float((s.outcome == "slip").mean())
        out.append(r)
    return out


def by_bins(p, df, ease_models):
    """Pooled over models and framings: matched hit/slip vs question ease and distractor pull.
    Ease = how many of `ease_models` (the 5 small models, which answered every question)
    solve it with no persona, so the scale is the same whichever models are on the page."""
    m = p[p.condition == "matched"].copy()
    nop = df[(df.condition == "none") & df.model.isin(ease_models)]
    solvers = nop.groupby("question_id").outcome.apply(lambda s: int((s == "correct").sum()))
    m["nsolve"] = m.question_id.map(solvers)
    ease = [{"bin": str(k), "n": len(g), "hit": float((g.outcome == "hit").mean()),
             "slip": float((g.outcome == "slip").mean())} for k, g in m.groupby("nsolve")]
    ctrl = (p[p.condition == "mismatched"].assign(h=lambda d: d.outcome == "hit")
            .groupby(["question_id", "target_letter"]).h.mean().rename("ctrl").reset_index())
    m = m.merge(ctrl, on=["question_id", "target_letter"])
    edges = [(-0.01, 0.0, "never"), (0.0, 0.1, "1–10%"), (0.1, 0.3, "11–30%"), (0.3, 1.0, "over 30%")]
    pull = []
    for lo, hi, name in edges:
        g = m[(m.ctrl > lo) & (m.ctrl <= hi)]
        pull.append({"bin": name, "n": len(g), "pairs": int(g[["question_id", "target_letter"]].drop_duplicates().shape[0]),
                     "hit": float((g.outcome == "hit").mean()), "slip": float((g.outcome == "slip").mean())})
    return ease, pull, m


def per_pair(m):
    """For each pair, out of (models x framings) matched tries: how many hit, how many were correct."""
    g = m.groupby(["question_id", "target_letter"]).agg(
        tries=("outcome", "size"), hits=("outcome", lambda s: int((s == "hit").sum())),
        slips=("outcome", lambda s: int((s == "slip").sum()))).reset_index()
    tries = int(g.tries.max())
    g = g[g.tries == tries]  # sample stage: one model may cover fewer pairs; keep pairs every model answered
    hist = [{"k": k, "hits": int((g.hits == k).sum()), "slips": int((g.slips == k).sum())} for k in range(tries + 1)]
    return g, hist, tries


def examples(df, g, tries, pairs, names, per_group, seed):
    """A few pairs per pattern, with every model's picks."""
    strong = {"claude-haiku-4-5", "gpt-4.1-mini", "gpt-5-mini"}
    weak = {"gpt-4o-mini", "gemini-2.5-flash-lite"}
    mm = df[(df.condition == "matched")]
    slip_by = (mm.assign(s=mm.outcome == "slip", h=mm.outcome == "hit")
               .groupby(["question_id", "target_letter", "model"])[["s", "h"]].mean().reset_index())
    sw = slip_by.pivot_table(index=["question_id", "target_letter"], columns="model", values=["s", "h"])
    leak = [k for k in sw.index if all(sw[("h", x)].get(k, 0) == 1 for x in strong if ("h", x) in sw)
            and all(sw[("s", x)].get(k, 0) == 1 for x in weak if ("s", x) in sw)]
    groups = {
        "truth": ("The correct answer always wins", g[g.slips == tries]),
        "all": ("Every model, both framings, made the tagged mistake", g[g.hits == tries]),
        "neither": ("Never correct, never the tagged mistake", g[(g.hits == 0) & (g.slips == 0)]),
        "leak": ("Strong models made the mistake, weak models answered correctly",
                 g.set_index(["question_id", "target_letter"]).loc[lambda x: x.index.isin(leak)].reset_index()),
    }
    by_q = {}
    for p_ in pairs:
        by_q.setdefault(p_["question_id"], []).append(p_)
    chosen, counts = {}, {}
    for key, (_, rows) in groups.items():
        counts[key] = len(rows)
        chosen[key] = rows.sample(min(per_group, len(rows)), random_state=seed)[["question_id", "target_letter"]].values.tolist()
    qids = {int(q) for v in chosen.values() for q, _ in v}
    questions = {}
    for q in qids:
        ps = by_q[q]
        questions[q] = {"text": ps[0]["question_text"], "options": ps[0]["options"], "correct": ps[0]["correct_letter"],
                        "subject": ps[0]["subject_name"], "construct": ps[0]["construct_name"],
                        "tags": {x["distractor_letter"]: x["misconception_name"] for x in ps}}
    sub = df[df.question_id.isin(qids)]
    answers, mismatch = {}, {}
    for r in sub.itertuples():
        a = answers.setdefault(r.model, {}).setdefault(int(r.question_id), {})
        cell = [r.picked_letter or "", CODE[r.outcome]]
        if r.condition == "none":
            a["none"] = cell
        else:
            a[f"{r.target_letter}|{r.framing}|{r.condition}"] = cell
            if r.condition == "mismatched":
                mismatch[f"{int(r.question_id)}|{r.target_letter}"] = names.get(r.persona_misconception_id, "")
    ex = {k: {"title": groups[k][0], "count": counts[k], "pairs": [[int(q), t] for q, t in v]} for k, v in chosen.items()}
    return ex, questions, answers, mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full")
    ap.add_argument("--examples", type=int, default=12, help="example pairs per pattern")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    df, p = load(cfg, args.stage)
    pairs = [json.loads(l) for l in open(ROOT / cfg["paths"]["processed_dir"] / "pairs.jsonl", encoding="utf-8")]
    names = {x["misconception_id"]: x["misconception_name"] for x in pairs}

    ease, pull, m = by_bins(p, df, cfg["full"]["models"])
    g, hist, tries = per_pair(m)
    ex, questions, answers, mismatch = examples(df, g, tries, pairs, names, args.examples, cfg["seed"])
    order = [x for x in cfg["models"] if x in set(df.model)]
    data = {"stage": args.stage, "models": order,
            "reasoning": {x: cfg["models"][x].get("reasoning", False) for x in order},
            "counts": {"pairs": int(m[["question_id", "target_letter"]].drop_duplicates().shape[0]), "common_pairs": int(g.shape[0]), "questions": int(df.question_id.nunique()), "rows": len(df)},
            "pairs_by_model": {x: int(m[m.model == x][["question_id", "target_letter"]].drop_duplicates().shape[0]) for x in order},
            "per_model": per_model(p), "ease": ease, "pull": pull, "hist": hist, "tries": tries,
            "examples": ex, "questions": questions, "answers": answers, "mismatch": mismatch}

    # Same look as the main report: reuse its stylesheet.
    style = re.search(r"<style>.*?</style>", (SRC / "report_template.html").read_text(encoding="utf-8"), re.S).group(0)
    html = ((SRC / "report_truth_template.html").read_text(encoding="utf-8")
            .replace("<!--__STYLE__-->", style)
            .replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=float).replace("</", "<\\/")))
    out = ROOT / "reports" / f"{args.stage}_truth.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(json.dumps({k: v["count"] for k, v in ex.items()}), f"\nwrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
