"""Phase 1 runner: matched / mismatched / no-persona conditions per pair.

Usage:
  python src/run_personas.py --stage pilot [--dry-run] [--yes]

Prints the call count and cost estimate first; runs above
budget.confirm_above_calls need --yes. Responses are cached (src/llm.py), so an
interrupted run resumes for free. Results store IDs only, no question text.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402
from llm import LLM, BudgetExceeded  # noqa: E402
from prompts import LETTERS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    proc = ROOT / cfg["paths"]["processed_dir"]
    pairs = [json.loads(line) for line in open(proc / "pairs.jsonl", encoding="utf-8")]
    holdout = set(json.load(open(proc / "holdout_question_ids.json")))
    return cfg, pairs, holdout


def select_pilot(pairs, holdout, n, seed):
    """n pairs from n distinct non-holdout questions."""
    by_q = defaultdict(list)
    for p in pairs:
        if p["question_id"] not in holdout:
            by_q[p["question_id"]].append(p)
    rng = random.Random(f"{seed}|pilot")
    qids = rng.sample(sorted(by_q), n)
    return [rng.choice(by_q[q]) for q in qids]


def select_full(pairs, seed):
    """All pairs, in a seeded order (the order drives the balanced target positions)."""
    out = sorted(pairs, key=lambda p: (p["question_id"], p["distractor_letter"]))
    random.Random(f"{seed}|full").shuffle(out)
    return out


def select_sample(pairs, seed, n):
    """The first n pairs of the full run's order. Same index -> same option order and
    control persona as the full run, so small models' sample answers come from the cache,
    and a smaller n for one model is a subset of everyone else's pairs."""
    return select_full(pairs, seed)[:n]


class Mismatcher:
    """A misconception not tagged on this question, from a different subject (else construct)."""

    def __init__(self, all_pairs, seed):
        self.seed = seed
        self.tagged = defaultdict(set)
        self.subjects, self.constructs, self.names = defaultdict(set), defaultdict(set), {}
        for p in all_pairs:
            self.tagged[p["question_id"]].add(p["misconception_id"])
            self.subjects[p["misconception_id"]].add(p["subject_id"])
            self.constructs[p["misconception_id"]].add(p["construct_id"])
            self.names[p["misconception_id"]] = p["misconception_name"]
        self.ids = sorted(self.names)

    def __call__(self, pair):
        here = self.tagged[pair["question_id"]]
        pool = [m for m in self.ids if m not in here and pair["subject_id"] not in self.subjects[m]]
        if not pool:
            pool = [m for m in self.ids if m not in here and pair["construct_id"] not in self.constructs[m]]
        rng = random.Random(f"{self.seed}|mismatch|{pair['question_id']}|{pair['distractor_letter']}")
        m = rng.choice(pool)
        return m, self.names[m]


def perm_for(seed, qid, anchor, pos, framing):
    """Display order as a list of original letters, with `anchor` placed at index `pos`
    and the other options shuffled (seeded). Same (question, anchor, framing) -> same order,
    so matched and mismatched personas see identical layouts."""
    rest = [L for L in LETTERS if L != anchor]
    random.Random(f"{seed}|shuffle|{qid}|{anchor}|{framing}").shuffle(rest)
    return rest[:pos] + [anchor] + rest[pos:]


def build_items(selected, all_pairs, framings, seed):
    """Target position is balanced: pair i, framing f puts the target at slot (i + f) % 4.
    The no-persona call balances the correct answer's slot the same way."""
    items, seen_q = [], set()
    mismatch = Mismatcher(all_pairs, seed)
    for i, p in enumerate(selected):
        mm_id, mm_name = mismatch(p)
        base = {"question_id": p["question_id"], "target_letter": p["distractor_letter"],
                "correct_letter": p["correct_letter"], "tagged_misconception_id": p["misconception_id"],
                "_q": p}
        for f, framing in enumerate(framings):
            perm = perm_for(seed, p["question_id"], p["distractor_letter"], (i + f) % 4, framing)
            for cond, mid, name in [("matched", p["misconception_id"], p["misconception_name"]),
                                    ("mismatched", mm_id, mm_name)]:
                items.append({**base, "condition": cond, "framing": framing,
                              "persona_misconception_id": mid, "_persona": name, "perm": perm})
        if p["question_id"] not in seen_q:
            seen_q.add(p["question_id"])
            items.append({**base, "target_letter": None, "tagged_misconception_id": None,
                          "condition": "none", "framing": "none",
                          "persona_misconception_id": None, "_persona": None,
                          "perm": perm_for(seed, p["question_id"], p["correct_letter"], i % 4, "none")})
    return items


def shown(item):
    return [item["_q"]["options"][L] for L in item["perm"]]


def batches(items, size):
    """Group by (framing, condition) so a batch never mixes matched/mismatched of one question."""
    groups = defaultdict(list)
    for it in items:
        groups[(it["framing"], it["condition"])].append(it)
    out = []
    for key in sorted(groups):
        g = groups[key]
        for i in range(0, len(g), size):
            chunk = g[i:i + size]
            assert len({it["question_id"] for it in chunk}) == len(chunk), "repeat question in batch"
            out.append(chunk)
    return out


def plan_calls(items, modes, batch_size):
    """[(mode, items_in_call, messages, max_tokens, response_format)] without calling anything."""
    calls = []
    if "single" in modes:
        for it in items:
            msgs = prompts.single_messages(it["framing"], it["_persona"], it["_q"]["question_text"], shown(it))
            calls.append(("single", [it], msgs, 20, prompts.ANSWER_SCHEMA))
    if "batched" in modes:
        for chunk in batches(items, batch_size):
            msgs = prompts.batched_messages(chunk[0]["framing"],
                                            [(it["_persona"], it["_q"]["question_text"], shown(it)) for it in chunk])
            calls.append(("batched", chunk, msgs, 8 * len(chunk) + 20, None))
    return calls


def estimate(llm, model_key, calls):
    """Input: ~3.5 chars/token (LaTeX-heavy) plus the structured-output overhead measured
    per model (est_schema_tokens). Output: ~15 tokens for {"answer": "X"}."""
    m = llm.cfg["models"][model_key]
    price = m["price_per_m"]
    todo = [c for c in calls if not llm.is_cached(model_key, c[2], c[3], c[4])]
    in_tok = sum(sum(len(x["content"]) for x in c[2]) / 3.5 + (m.get("est_schema_tokens", 35) if c[4] else 0)
                 for c in todo)
    out_tok = sum((15 if c[4] else c[3]) + m.get("est_hidden_tokens", 0) for c in todo)
    usd = (in_tok * price["input"] + out_tok * price["output"]) / 1e6
    return len(calls), len(todo), in_tok, usd


def outcome(item, orig):
    if orig is None:
        return "invalid"
    if item["condition"] == "none":
        return "correct" if orig == item["correct_letter"] else "incorrect"
    if orig == item["target_letter"]:
        return "hit"
    return "slip" if orig == item["correct_letter"] else "wrong_stupid"


def record(item, model_key, mcfg, mode, raw, shown_letter, retried, cached, batch_id=None, pos=None):
    orig = item["perm"][LETTERS.index(shown_letter)] if shown_letter else None
    row = {k: v for k, v in item.items() if not k.startswith("_")}
    row.update({"model": model_key, "reasoning": mcfg.get("reasoning", False), "mode": mode,
                "batch_id": batch_id, "batch_pos": pos, "raw": raw, "shown_letter": shown_letter,
                "picked_letter": orig, "outcome": outcome(item, orig), "retried": retried, "cached": cached})
    return row


def run_call(llm, model_key, bid, call):
    """One API call (plus at most one format retry) -> result rows for its items."""
    mcfg = llm.cfg["models"][model_key]
    mode, its, msgs, max_tok, rf = call
    try:
        r = llm.chat(model_key, msgs, max_tok, rf)
    except openai.BadRequestError as e:
        # e.g. a content filter on one question: log as invalid, don't kill a long run.
        # Not cached, so a rerun tries again.
        print(f"  call {bid}: BadRequest {str(e)[:120]}")
        return [record(it, model_key, mcfg, mode, [f"ERROR: {str(e)[:200]}"], None,
                       False, False, bid if mode == "batched" else None, pos)
                for pos, it in enumerate(its, 1)]
    raw, retried, cached = [r["text"]], False, r["cached"]
    if mode == "single":
        letter = prompts.parse_single(r["text"])
        if letter is None:
            retry = msgs + [{"role": "assistant", "content": r["text"]},
                            {"role": "user", "content": prompts.RETRY_SINGLE}]
            r2 = llm.chat(model_key, retry, max_tok, rf)
            raw.append(r2["text"]); retried = True
            letter = prompts.parse_single(r2["text"])
        return [record(its[0], model_key, mcfg, mode, raw, letter, retried, cached)]
    got = prompts.parse_batched(r["text"], len(its))
    if len(got) < len(its):
        retry = msgs + [{"role": "assistant", "content": r["text"]},
                        {"role": "user", "content": prompts.RETRY_BATCHED}]
        r2 = llm.chat(model_key, retry, max_tok)
        raw.append(r2["text"]); retried = True
        for i, L in prompts.parse_batched(r2["text"], len(its)).items():
            got.setdefault(i, L)
    return [record(it, model_key, mcfg, mode, raw, got.get(pos), retried, cached, bid, pos)
            for pos, it in enumerate(its, 1)]


def execute(llm, model_key, calls, workers=1):
    """Run calls with `workers` concurrent requests; rows come back in call order."""
    every = max(25, len(calls) // 40)
    done = [0]

    def task(arg):
        out = run_call(llm, model_key, *arg)
        with llm.lock:
            done[0] += 1
            if done[0] % every == 0:
                print(f"  {done[0]}/{len(calls)} calls  (new: {llm.session['calls']}, "
                      f"cached: {llm.session['cached']}, ${llm.session['usd']:.4f})")
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            results = list(pool.map(task, enumerate(calls)))
        except BudgetExceeded:
            pool.shutdown(cancel_futures=True)
            raise
    return [row for rows in results for row in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "full", "sample"], default="pilot")
    ap.add_argument("--model", help="model key from config (one process per model; pilot defaults to pilot.model)")
    ap.add_argument("--dry-run", action="store_true", help="print the estimate and stop")
    ap.add_argument("--yes", action="store_true", help="confirm runs above the call threshold")
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests for this model")
    args = ap.parse_args()

    cfg, pairs, holdout = load()
    seed, st = cfg["seed"], cfg[args.stage]
    model_key = args.model or st.get("model")
    if model_key not in cfg["models"]:
        ap.error(f"--model must be one of {list(cfg['models'])}")
    if args.stage == "pilot":
        selected = select_pilot(pairs, holdout, st["n_pairs"], seed)
    elif args.stage == "sample":
        selected = select_sample(pairs, seed, st.get("n_pairs_by_model", {}).get(model_key, st["n_pairs"]))
    else:
        selected = select_full(pairs, seed)
    out_name = f"{args.stage}_{model_key}_responses.jsonl"
    items = build_items(selected, pairs, st["framings"], seed)
    calls = plan_calls(items, st["modes"], st["batch_size"])

    llm = LLM(cfg)
    n, todo, in_tok, usd = estimate(llm, model_key, calls)
    by_mode = defaultdict(int)
    for c in calls:
        by_mode[c[0]] += 1
    print(f"stage={args.stage} model={model_key} seed={seed} pairs={len(selected)} items={len(items)} workers={args.workers}")
    print(f"calls: {n} {dict(by_mode)}  not cached: {todo}  (+ up to {todo} retries if replies don't parse)")
    print(f"est. input tokens: {in_tok:,.0f}   est. cost: ${usd:.4f}   "
          f"spent today: ${llm.spent_today():.4f} / cap ${cfg['budget']['daily_usd_cap']}")
    if args.dry_run:
        return
    if todo > cfg["budget"]["confirm_above_calls"] and not args.yes:
        print(f"More than {cfg['budget']['confirm_above_calls']} new calls: review the estimate, then rerun with --yes.")
        return

    try:
        rows = execute(llm, model_key, calls, args.workers)
    except BudgetExceeded as e:
        print(f"STOPPED: {e}. Rerun tomorrow; cached calls are free.")
        return
    out = ROOT / cfg["paths"]["results_dir"] / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"done: {llm.session['calls']} new calls, {llm.session['cached']} cached, "
          f"${llm.session['usd']:.4f}. wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
