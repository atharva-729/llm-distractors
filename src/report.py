"""Build a self-contained local HTML report from Phase 1 results.

Usage: python src/report.py --stage full   ->  reports/full.html

The report embeds real Eedi question text, so it lives in reports/ (gitignored)
and is for local viewing only, not the public site (see LICENSE_NOTES.md).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "report_template.html"


def outcome_mix(df):
    m = df[df.condition == "matched"]
    out = []
    for (model, framing), g in m.groupby(["model", "framing"]):
        out.append({"model": model, "framing": framing, "n": len(g),
                    **{o: float((g.outcome == o).mean()) for o in ["hit", "slip", "wrong_stupid", "invalid"]}})
    return out


def consistency(df, min_q):
    """Per misconception tagged in >= min_q questions: share of its matched pairs that hit."""
    m = df[df.condition == "matched"]
    nq = m.groupby(["model", "framing", "tagged_misconception_id"]).question_id.nunique()
    hit = m.groupby(["model", "framing", "tagged_misconception_id"]).outcome.apply(lambda s: (s == "hit").mean())
    t = pd.DataFrame({"nq": nq, "hit": hit}).reset_index()
    t = t[t.nq >= min_q]
    out = []
    for (model, framing), g in t.groupby(["model", "framing"]):
        out.append({"model": model, "framing": framing, "values": g.hit.round(4).tolist(),
                    "n": len(g), "median": float(g.hit.median()),
                    "always": float((g.hit == 1).mean()), "never": float((g.hit == 0).mean())})
    return out


def by_subject(df, top):
    p = df[df.condition != "none"]
    sizes = p[(p.condition == "matched")].groupby("subject").size()
    keep = sizes.sort_values(ascending=False).head(top).index
    out = []
    for (model, framing, subj), g in p[p.subject.isin(keep)].groupby(["model", "framing", "subject"]):
        mh = (g[g.condition == "matched"].outcome == "hit").mean()
        mm = (g[g.condition == "mismatched"].outcome == "hit").mean()
        out.append({"model": model, "framing": framing, "subject": subj,
                    "n": int((g.condition == "matched").sum()), "lift": float(mh - mm), "hit": float(mh)})
    return out, list(keep)


def examples(df, pairs, names):
    """Questions + every model's answers, keyed for the example browser."""
    by_q = defaultdict(list)
    for p in pairs:
        by_q[p["question_id"]].append(p)
    qids = sorted(df.question_id.unique())
    questions = {}
    for q in qids:
        ps = by_q[q]
        questions[int(q)] = {
            "text": ps[0]["question_text"], "options": ps[0]["options"], "correct": ps[0]["correct_letter"],
            "subject": ps[0]["subject_name"], "construct": ps[0]["construct_name"],
            "tags": {p["distractor_letter"]: p["misconception_name"] for p in ps}}
    # Compact cells: [picked (original letter), outcome code, shown letter, display order, raw if unusual]
    code = {"hit": "h", "slip": "s", "wrong_stupid": "w", "invalid": "i", "correct": "c", "incorrect": "x"}
    answers = defaultdict(lambda: defaultdict(dict))
    mismatch = {}
    for r in df.itertuples():
        q = int(r.question_id)
        raw = r.raw[-1] if isinstance(r.raw, list) and r.raw else ""
        standard = isinstance(r.shown_letter, str) and raw.replace(" ", "").replace("\n", "") == f'{{"answer":"{r.shown_letter}"}}'
        cell = [r.picked_letter or "", code[r.outcome], r.shown_letter or "", "".join(r.perm),
                "" if standard else raw[:80] + (" (after retry)" if len(r.raw) > 1 else "")]
        if r.condition == "none":
            answers[r.model][q]["none"] = cell
        else:
            answers[r.model][q][f"{r.target_letter}|{r.framing}|{r.condition}"] = cell
            if r.condition == "mismatched":
                mismatch[f"{q}|{r.target_letter}"] = names.get(r.persona_misconception_id, "")
    return questions, {m: dict(v) for m, v in answers.items()}, mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full")
    ap.add_argument("--subjects", type=int, default=12)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    res = ROOT / cfg["paths"]["results_dir"]
    files = sorted(res.glob(f"{args.stage}_*_responses.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df = df[df["mode"] == "single"]
    pairs = [json.loads(l) for l in open(ROOT / cfg["paths"]["processed_dir"] / "pairs.jsonl", encoding="utf-8")]
    names = {p["misconception_id"]: p["misconception_name"] for p in pairs}
    subj = {p["question_id"]: p["subject_name"] for p in pairs}
    df["subject"] = df.question_id.map(subj)

    order = [m for m in cfg["models"] if m in set(df.model)]
    met = metrics(df)
    leaderboard = met[["model", "framing", "matched_n", "matched_hit_rate", "mismatched_hit_rate", "lift",
                       "solvable_n", "solvable_lift", "no_persona_accuracy"]].to_dict("records")
    subjects, subject_order = by_subject(df, args.subjects)
    questions, answers, mismatch = examples(df, pairs, names)
    data = {
        "stage": args.stage, "models": order,
        "reasoning": {m: cfg["models"][m].get("reasoning", False) for m in order},
        "counts": {"pairs": int(df[df.condition == "matched"].drop_duplicates(["question_id", "target_letter"]).shape[0]),
                   "questions": int(df.question_id.nunique()), "rows": len(df)},
        "leaderboard": leaderboard, "mix": outcome_mix(df),
        "consistency": consistency(df, cfg["data"]["consistency_min_questions"]),
        "subjects": subjects, "subject_order": subject_order,
        "questions": questions, "answers": answers, "mismatch": mismatch,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=float).replace("</", "<\\/"))
    out = ROOT / "reports" / f"{args.stage}.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"models: {order}\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
