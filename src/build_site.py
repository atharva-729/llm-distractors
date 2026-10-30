"""Build the public GitHub Page (docs/index.html) from Phase 1 results.

Usage: python src/build_site.py

Public page rules (LICENSE_NOTES.md): aggregates only, no Eedi question text or
misconception names. The worked examples are our own questions (src/site_examples.py).
Main charts use the pairs every model answered (the first 250 of the seeded sample);
detail charts add the full 2,734-pair run of the five small models.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_examples import QUESTIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
KEY = ["question_id", "target_letter"]

# Display order, names and tiers for the page.
MODELS = [
    ("gpt-4o-mini", "GPT-4o mini", "small"),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", "small"),
    ("gpt-5-mini", "GPT-5 mini", "small"),
    ("gpt-4.1-mini", "GPT-4.1 mini", "small"),
    ("claude-haiku-4-5", "Claude Haiku 4.5", "small"),
    ("gpt-5.4", "GPT-5.4", "large"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6", "large"),
    ("gpt-5.4-think", "GPT-5.4 + thinking", "thinking"),
    ("gemini-3.1-pro", "Gemini 3.1 Pro", "thinking"),
]
SMALL = [m for m, _, t in MODELS if t == "small"]
PANEL = 25  # personas per audited question in Phase 2 (README)


def read(stage, model):
    return pd.read_json(ROOT / "results" / f"{stage}_{model}_responses.jsonl", lines=True)


def boot(rng, arrays, fn, n=2000):
    """95% interval of fn over pair-level resamples (arrays share the pair index)."""
    k = len(arrays[0])
    idx = rng.integers(0, k, size=(n, k))
    vals = np.array([fn(*[a[i] for a in arrays]) for i in idx])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pair_table(df, keys=None):
    """One row per (pair, framing): matched and control outcomes side by side."""
    p = df[df.condition != "none"]
    w = p.pivot_table(index=KEY + ["framing"], columns="condition", values="outcome", aggfunc="first").reset_index()
    if keys is not None:
        w = w.merge(keys, on=KEY)
    return w


def summary(df, keys, qids, rng):
    out = {}
    w = pair_table(df, keys)
    alone = df[(df.condition == "none") & df.question_id.isin(qids)].outcome.eq("correct").to_numpy(float)
    for fr, g in w.groupby("framing"):
        m, c = g.matched.to_numpy(), g.mismatched.to_numpy()
        hm, hc = (m == "hit").astype(float), (c == "hit").astype(float)
        sm, sc = (m == "slip").astype(float), (c == "slip").astype(float)
        miss = (m == "slip") | (m == "wrong_stupid")
        out[fr] = {
            "n": len(g), "hit_m": hm.mean(), "hit_c": hc.mean(), "lift": hm.mean() - hc.mean(),
            "slip_m": sm.mean(), "other_m": (m == "wrong_stupid").mean(), "inv_m": (m == "invalid").mean(),
            "correct_c": sc.mean(), "other_c": (c == "wrong_stupid").mean(),
            "misses": int(miss.sum()), "truth_share": float(sm[miss].mean()) if miss.any() else None,
            "ci_hit_m": boot(rng, [hm], np.mean), "ci_hit_c": boot(rng, [hc], np.mean),
            "ci_lift": boot(rng, [hm, hc], lambda a, b: a.mean() - b.mean()),
            "ci_correct_c": boot(rng, [sc], np.mean), "ci_slip_m": boot(rng, [sm], np.mean),
        }
    out["alone"] = float(alone.mean())
    out["ci_alone"] = boot(rng, [alone], np.mean)
    return out


def cost_per_call(cfg, model):
    """Mean measured cost of a first-attempt persona call (JSON answer mode)."""
    rows = []
    for line in open(ROOT / cfg["paths"]["cache_dir"] / f"llm_cache_{model}.jsonl", encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "response_format" in r["params"] and len(r["messages"]) == 2:
            rows.append(r["usd"])
    return float(np.mean(rows))


def flows(a, b, keys):
    """Outcome of the same item without (a) and with (b) thinking."""
    k = KEY + ["framing", "condition"]
    m = a[a.condition != "none"].merge(b[b.condition != "none"], on=k, suffixes=("_a", "_b")).merge(keys, on=KEY)
    out = {}
    for cond, g in m.groupby("condition"):
        out[cond] = [{"from": f, "to": t, "n": int(n)} for (f, t), n in g.groupby(["outcome_a", "outcome_b"]).size().items()]
    return out


def natural(sample, keys):
    """Pooled over models: matched hit / correct by how often control personas picked that option."""
    p = pd.concat(sample.values())
    p = p[p.condition != "none"].merge(keys, on=KEY)
    ctrl = p[p.condition == "mismatched"].assign(h=lambda d: d.outcome == "hit").groupby(KEY).h.mean().rename("ctrl")
    m = p[p.condition == "matched"].merge(ctrl.reset_index(), on=KEY)
    out = []
    for lo, hi, name in [(-1, 0, "never"), (0, .1, "1–10%"), (.1, .3, "11–30%"), (.3, 1, "over 30%")]:
        g = m[(m.ctrl > lo) & (m.ctrl <= hi)]
        out.append({"bin": name, "pairs": int(g[KEY].drop_duplicates().shape[0]), "n": len(g),
                    "hit": float((g.outcome == "hit").mean()), "slip": float((g.outcome == "slip").mean())})
    return out


def agreement(sample, keys):
    p = pd.concat(sample.values())
    m = p[p.condition == "matched"].merge(keys, on=KEY)
    g = m.groupby(KEY).agg(tries=("outcome", "size"), hits=("outcome", lambda s: int((s == "hit").sum())),
                           slips=("outcome", lambda s: int((s == "slip").sum())))
    t = int(g.tries.max())
    return {"tries": t, "hist": [{"k": k, "hits": int((g.hits == k).sum()), "slips": int((g.slips == k).sum())}
                                 for k in range(t + 1)]}


def full_run(rng):
    out, frames = {}, []
    for m in SMALL:
        d = read("full", m)
        frames.append(d)
        w = pair_table(d)
        out[m] = {}
        for fr, g in w.groupby("framing"):
            hm, hc = (g.matched == "hit").to_numpy(float), (g.mismatched == "hit").to_numpy(float)
            out[m][fr] = {"n": len(g), "hit_m": hm.mean(), "hit_c": hc.mean(), "lift": hm.mean() - hc.mean(),
                          "ci_lift": boot(rng, [hm, hc], lambda a, b: a.mean() - b.mean(), n=1000)}
    return out, pd.concat(frames)


def by_subject(full, pairs, top=10):
    subj = {p["question_id"]: p["subject_name"] for p in pairs}
    p = full[full.condition != "none"].assign(subject=lambda d: d.question_id.map(subj))
    sizes = p[(p.condition == "matched") & (p.model == SMALL[0]) & (p.framing == "student")].groupby("subject").size()
    keep = list(sizes.sort_values(ascending=False).head(top).index)
    cells = []
    for (m, fr, s), g in p[p.subject.isin(keep)].groupby(["model", "framing", "subject"]):
        hm = (g[g.condition == "matched"].outcome == "hit").mean()
        hc = (g[g.condition == "mismatched"].outcome == "hit").mean()
        cells.append({"model": m, "framing": fr, "subject": s, "lift": float(hm - hc), "hit": float(hm),
                      "n": int((g.condition == "matched").sum())})
    return {"subjects": [{"name": s, "n": int(sizes[s])} for s in keep], "cells": cells}


def consistency(full, min_q):
    m = full[full.condition == "matched"]
    g = m.groupby(["model", "framing", "tagged_misconception_id"]).agg(
        nq=("question_id", "nunique"), hit=("outcome", lambda s: (s == "hit").mean())).reset_index()
    g = g[g.nq >= min_q]
    edges = [("never", lambda v: v == 0), ("1–33%", lambda v: (v > 0) & (v <= 1 / 3 + 1e-9)),
             ("34–66%", lambda v: (v > 1 / 3 + 1e-9) & (v < 2 / 3 - 1e-9)),
             ("67–99%", lambda v: (v >= 2 / 3 - 1e-9) & (v < 1)), ("always", lambda v: v == 1)]
    out = {}
    for (model, fr), h in g.groupby(["model", "framing"]):
        out.setdefault(model, {})[fr] = {"n": len(h), "bins": [int(fn(h.hit).sum()) for _, fn in edges],
                                         "median": float(h.hit.median())}
    return {"labels": [e for e, _ in edges], "data": out}


def examples():
    rows = [json.loads(l) for l in open(ROOT / "results" / "site_examples.jsonl", encoding="utf-8")]
    out = []
    for q in QUESTIONS:
        ans = {}
        for r in rows:
            if r["question"] != q["id"]:
                continue
            k = "none" if r["condition"] == "none" else f"{r['target']}|{r['condition']}"
            ans.setdefault(r["model"], {})[k] = r["pick"]
        out.append({**q, "answers": ans})
    return out


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    rng = np.random.default_rng(cfg["seed"])
    pairs = [json.loads(l) for l in open(ROOT / cfg["paths"]["processed_dir"] / "pairs.jsonl", encoding="utf-8")]
    sample = {m: read("sample", m) for m, _, _ in MODELS}
    # Pairs every model answered (the smallest per-model sample is a prefix of the others).
    keys = None
    for d in sample.values():
        k = d[d.condition == "matched"][KEY].drop_duplicates()
        keys = k if keys is None else keys.merge(k, on=KEY)
    qids = set(keys.question_id)

    models = []
    for m, name, tier in MODELS:
        c = cost_per_call(cfg, m)
        models.append({"key": m, "name": name, "tier": tier, "cost_call": c,
                       "cost_1000q": c * PANEL * 1000, **summary(sample[m], keys, qids, rng)})
    full, full_df = full_run(rng)
    answers = sum(len(read("full", m)) for m in SMALL) + sum(len(sample[m]) for m, _, t in MODELS if t != "small")
    data = {
        "counts": {"answers": answers, "eedi_questions": 1869, "pairs": len(pairs), "questions": len({p["question_id"] for p in pairs}),
                   "misconceptions": len({p["misconception_id"] for p in pairs}), "catalogue": 2587,
                   "common_pairs": len(keys), "common_questions": len(qids), "panel": PANEL},
        "models": models,
        "flows": flows(sample["gpt-5.4"], sample["gpt-5.4-think"], keys),
        "natural": natural(sample, keys), "agreement": agreement(sample, keys),
        "full": full, "subjects": by_subject(full_df, pairs),
        "consistency": consistency(full_df, cfg["data"]["consistency_min_questions"]),
        "examples": examples(),
    }
    html = (SRC / "site_template.html").read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=float).replace("</", "<\\/"))
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    (out.parent / ".nojekyll").write_text("")
    # Guard: the public page must not contain Eedi question text or misconception names.
    names = set(pd.read_csv(ROOT / cfg["paths"]["raw_dir"] / "misconception_mapping.csv").iloc[:, 1].astype(str))
    leaked = [n for n in names if len(n) > 25 and n in html]
    assert not leaked, f"Eedi misconception names on the public page: {leaked[:3]}"
    print(f"common pairs {len(keys)} / questions {len(qids)}; wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
