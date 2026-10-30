"""Phase 0: build data/processed/pairs.jsonl from the raw Eedi Kaggle files.

One row per tagged distractor. Question/option text is only whitespace- and
encoding-normalised; LaTeX is left as-is.

Usage: python src/data_prep.py [--show N]
"""
import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
LETTERS = ["A", "B", "C", "D"]
IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean(text):
    """Whitespace/encoding fixes only: NFC, unify newlines, strip trailing spaces."""
    if pd.isna(text):
        return text
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def has_image(row):
    fields = [row["QuestionText"]] + [row[f"Answer{L}Text"] for L in LETTERS]
    return any(isinstance(s, str) and IMAGE_RE.search(s) for s in fields)


def build_pairs(train, names, drop_images):
    rows, skipped = [], {"untagged": 0, "image": 0, "tag_on_correct": 0}
    for _, r in train.iterrows():
        image = has_image(r)
        options = {L: r[f"Answer{L}Text"] for L in LETTERS}
        for L in LETTERS:
            if L == r["CorrectAnswer"]:
                if pd.notna(r[f"Misconception{L}Id"]):
                    skipped["tag_on_correct"] += 1
                continue
            mid = r[f"Misconception{L}Id"]
            if pd.isna(mid):
                skipped["untagged"] += 1
                continue
            if image and drop_images:
                skipped["image"] += 1
                continue
            mid = int(mid)
            rows.append({
                "question_id": int(r["QuestionId"]),
                "question_text": r["QuestionText"],
                "options": options,
                "correct_letter": r["CorrectAnswer"],
                "distractor_letter": L,
                "misconception_id": mid,
                "misconception_name": names[mid],
                "construct_id": int(r["ConstructId"]),
                "construct_name": r["ConstructName"],
                "subject_id": int(r["SubjectId"]),
                "subject_name": r["SubjectName"],
                "has_image": image,
            })
    return rows, skipped


def make_holdout(pairs, n, seed):
    """Seeded sample of questions with all 3 distractors tagged (Phase 2 junk-swap)."""
    per_q = pd.DataFrame(pairs).groupby("question_id").size()
    full = sorted(per_q[per_q == 3].index.tolist())
    return sorted(random.Random(seed).sample(full, n))


def summarize(pairs, train, n_images, min_q):
    df = pd.DataFrame(pairs)
    per_mc = df.groupby("misconception_id")["question_id"].nunique()
    per_q = df.groupby("question_id").size()
    print("=== Raw ===")
    print(f"questions in train.csv:              {len(train)}")
    print(f"questions with images:               {n_images}")
    print("=== pairs.jsonl ===")
    print(f"questions:                           {df.question_id.nunique()}")
    print(f"pairs (tagged distractors):          {len(df)}")
    print(f"unique misconceptions:               {df.misconception_id.nunique()}")
    print(f"misconceptions in >= {min_q} questions:     {(per_mc >= min_q).sum()}"
          f"  (covering {df[df.misconception_id.isin(per_mc[per_mc >= min_q].index)].shape[0]} pairs)")
    print(f"questions with all 3 distractors tagged: {(per_q == 3).sum()}")
    print(f"tagged distractors per question:     {per_q.value_counts().sort_index().to_dict()}")
    print(f"subjects: {df.subject_name.nunique()}   constructs: {df.construct_name.nunique()}")
    print("misconception frequency (#questions -> #misconceptions):",
          per_mc.value_counts().sort_index().head(10).to_dict(), "...")
    print("top 10 subjects by pairs:")
    print(df.subject_name.value_counts().head(10).to_string())


def show(pairs, n, seed):
    rng = random.Random(seed)
    for p in rng.sample(pairs, n):
        print("\n" + "-" * 70)
        print(f"Q{p['question_id']}  [{p['subject_name']} / {p['construct_name']}]")
        print(p["question_text"])
        for L, txt in p["options"].items():
            tag = " (correct)" if L == p["correct_letter"] else ""
            tag += "  <-- distractor" if L == p["distractor_letter"] else ""
            print(f"  {L}) {txt}{tag}")
        print(f"Misconception {p['misconception_id']}: {p['misconception_name']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=5, help="random pairs to print")
    args = ap.parse_args()

    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw_dir"]
    out_dir = ROOT / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(raw / "train.csv")
    mapping = pd.read_csv(raw / "misconception_mapping.csv")
    text_cols = ["QuestionText", "ConstructName", "SubjectName"] + [f"Answer{L}Text" for L in LETTERS]
    for c in text_cols:
        train[c] = train[c].map(clean)
    mapping["MisconceptionName"] = mapping["MisconceptionName"].map(clean)
    names = dict(zip(mapping.MisconceptionId, mapping.MisconceptionName))

    n_images = int(train.apply(has_image, axis=1).sum())
    pairs, skipped = build_pairs(train, names, cfg["data"]["drop_image_questions"])

    with open(out_dir / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    holdout = make_holdout(pairs, cfg["data"]["holdout_questions"], cfg["seed"])
    with open(out_dir / "holdout_question_ids.json", "w", encoding="utf-8") as f:
        json.dump(holdout, f)

    print(f"seed={cfg['seed']}  drop_image_questions={cfg['data']['drop_image_questions']}")
    print(f"skipped distractors: {skipped}")
    summarize(pairs, train, n_images, cfg["data"]["consistency_min_questions"])
    print(f"\nwrote {len(pairs)} pairs -> {out_dir / 'pairs.jsonl'}")
    print(f"wrote {len(holdout)} holdout question ids -> {out_dir / 'holdout_question_ids.json'}")
    if args.show:
        show(pairs, args.show, cfg["seed"])


if __name__ == "__main__":
    main()
