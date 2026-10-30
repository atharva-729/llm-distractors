# Confused on Purpose — Misconception Personas for MCQ Distractors

## What this project is

Multiple-choice math questions have one correct answer and a few wrong ones (**distractors**). Good distractors aren't random: each one is the answer a student gets if they hold a specific wrong belief (a **misconception**). Example: for `1/2 + 1/3`, the distractor `2/5` catches students who "add the tops and add the bottoms."

A **persona** is an LLM prompted to be a student who holds exactly one misconception and is otherwise normal.

This project asks two questions:

1. **Experiment (Phase 1):** Can LLMs be confused *on purpose* — consistently making the specific mistake a given misconception produces, instead of slipping into the correct answer or making a random wrong one?
2. **Tool (Phase 2, only if Phase 1 works):** If personas are reliable, can a panel of them audit a question — flagging **dead distractors** (options no confused student would pick) and suggesting replacements that target uncovered misconceptions?

Data: the **Eedi – Mining Misconceptions in Mathematics** Kaggle dataset, where distractors are already tagged with misconceptions. **No manual labeling anywhere in this project.**

Final deliverable: a static GitHub Page with the persona leaderboard, findings, and audit-tool walkthroughs.

---

## Ground rules (read before every phase)

**Do**
- Keep everything simple and scriptable: Python, pandas, plain JSON/JSONL files. No databases, no web frameworks for the pipeline.
- Cache every LLM response to disk (JSONL, keyed by a hash of model + prompt). Runs must be resumable and never re-pay for a call already made.
- Put models, temperatures, sample sizes and paths in one `config.yaml`. No model names hardcoded in scripts.
- Use a provider-agnostic client (e.g. LiteLLM or OpenRouter) so models can be swapped from config.
- Fix random seeds (option shuffling, sampling) and log them.
- Print an estimated call count before any run larger than the pilot, and stop for confirmation.

**Don't**
- Don't fine-tune anything.
- Don't build Phase 2 before Phase 1's numbers are reviewed at the checkpoint.
- Don't invent column names or dataset structure — inspect the files first.
- Don't put API keys anywhere in the repo or the GitHub Page.
- Don't publish raw Eedi data on the page until the license has been checked (Phase 0).
- Don't add subjects beyond math, image-based questions, or a live backend.

---

## Repo structure

```
.
├── config.yaml
├── data/
│   ├── raw/            # Kaggle files, gitignored
│   └── processed/      # cleaned pairs, splits
├── src/
│   ├── data_prep.py
│   ├── llm.py          # client wrapper + disk cache
│   ├── prompts.py      # persona prompt templates
│   ├── run_personas.py # Phase 1 runner
│   ├── analyze.py      # Phase 1 metrics + plots
│   ├── panel.py        # Phase 2 audit tool
│   └── eval_audit.py   # Phase 2 junk-swap evaluation
├── results/            # JSONL responses, metric CSVs, figures
└── site/               # static GitHub Page
```

---

## Phase 0 — Data setup

**Goal:** a clean table of (question, distractor, misconception) pairs.

Steps:
1. Download the Eedi Kaggle competition files into `data/raw/` (requires accepting competition rules on Kaggle).
2. Inspect the files and print their actual columns. Expected (verify, don't assume): a training file with question text, four answer texts, the correct answer letter, and one misconception ID per answer option; plus a mapping file from misconception ID to a readable misconception name.
3. Build `data/processed/pairs.jsonl` — one row per **tagged distractor**:
   `question_id, question_text, options{A..D}, correct_letter, distractor_letter, misconception_id, misconception_name, subject/construct fields if present`
4. Drop distractors with no misconception tag. Drop questions that depend on images (if detectable).
5. Print summary stats: number of questions, number of pairs, number of unique misconceptions, how many misconceptions appear in ≥3 questions (needed for the consistency test).
6. Read the competition's data-use rules and write a short `LICENSE_NOTES.md`: can we show example questions publicly? Can we use it in a company demo?

**Don't:** clean or rewrite question text beyond whitespace/encoding fixes — the LaTeX should stay as-is.

### ✅ Checkpoint 0
Show Atharva: the summary stats, 5 random pairs printed in readable form, and `LICENSE_NOTES.md`. Wait for approval.

---

## Phase 1 — The experiment: can LLMs be confused on purpose?

**Goal:** measure, per model and prompt framing, how often a persona makes the *right* mistake.

### 1a. Prompts (`src/prompts.py`)
Two framings, each filled with the misconception name:
- **Student framing:** "You are a student who is normal in every way except you firmly believe: {misconception}. Answer the question as this student would. Reply with one letter only."
- **Teacher framing:** "A student holds this misconception: {misconception}. Predict which option this student would pick. Reply with one letter only."

Plus a **no-persona** prompt: just answer the question, one letter.

### 1b. Conditions per pair
For each (question, tagged distractor) pair:
- **Matched persona:** persona holds this distractor's misconception.
- **Mismatched persona (control):** persona holds a random misconception that is **not** tagged on any option of this question (prefer one from a different construct/topic).
- **No persona:** once per question, not per pair.

Shuffle option order per call (seeded) and map the answer letter back to the original option.

### 1c. Scoring
Each persona answer is one of:
- **Hit** — picked the distractor tagged with the persona's misconception
- **Slip** — picked the correct answer
- **Wrong-stupid** — picked a different distractor
- **Invalid** — unparseable reply (retry once, then log)

Metrics per (model, framing):
- hit / slip / wrong-stupid rates on matched personas
- mismatched "hit" rate (how often the control persona picks that same distractor anyway)
- **lift = matched hit rate − mismatched hit rate** ← the headline number; a high hit rate with no lift means the model just picks the most tempting distractor regardless of persona
- **consistency:** for misconceptions appearing in ≥3 questions, fraction of those questions where the persona hits; report the distribution
- breakdowns by subject/construct

### 1d. Run order
1. **Pilot:** 50 pairs × 1 cheap model × both framings. Check parsing, shuffling, and that scoring looks sane by reading ~10 raw responses.
2. **Full run:** all pairs (or a seeded sample if too expensive — decide at the pilot checkpoint) × 3–4 models (mix of cheap and frontier, from config) × both framings.

Temperature 0 where supported. If a model is a reasoning model, record that in results.

### 1e. Analysis (`src/analyze.py`)
- `results/phase1_metrics.csv`
- Plots: lift by model × framing; hit/slip/wrong-stupid stacked bars; consistency histogram
- A file of ~20 interesting raw examples: confident slips, perfect consistency streaks, weird wrong-stupid picks

**Don't:** tune prompts on the full dataset. If you change prompts, change them during the pilot only, then freeze.

### ✅ Checkpoint 1a (after pilot)
Show: 10 raw responses with scoring, parse failure rate, estimated call count and rough cost for the full run. Wait for approval before the full run.

### ✅ Checkpoint 1b (after full run) — DECISION GATE
Show: metrics table, plots, examples. Atharva decides:
- **Clear lift for at least one model/framing** (rough guide: matched hit rate well above the mismatched control, e.g. 15+ points) → go to Phase 2 using the best model + framing.
- **No meaningful lift** → skip Phase 2. The finding ("LLMs can't convincingly fake a specific confusion") becomes the main result; go straight to Phase 3.

---

## Phase 2 — The audit tool (only if the gate says go)

**Goal:** given a question, use a persona panel to flag dead distractors and suggest better ones.

### 2a. Panel selection (`src/panel.py`)
- Embed all misconception names once (any small embedding model from config).
- For a question, embed its text (+ construct/subject name if available) and take the top-k most similar misconceptions as the panel (k from config, start with 25).
- Sanity metric: on Eedi questions, what fraction of the question's true tagged misconceptions land in the top-k? Report it. If recall is poor, try including the answer options in the query text.

### 2b. Run the panel
- Run every panel persona on the question with the Phase 1 winning model + framing.
- Per option, count picks.

### 2c. Flags
- **Dead distractor:** a distractor picked by 0 (or ≤ threshold from config) personas.
- **Leaky key:** the correct answer picked by more than X% of personas (they are supposed to be confused) → the question may not separate understanding from misconception.

### 2d. Suggestions
For panel misconceptions that no current option captures:
1. Ask the model what answer a student with that misconception would produce for this question.
2. Drop suggestions that equal the correct answer or an existing option.
3. **Verify:** add the suggestion as an option and re-run that one persona; keep it only if the persona picks it.
4. Return the top 3 verified suggestions with the misconception each targets.

### 2e. Evaluation without labeling — the junk-swap test (`src/eval_audit.py`)
1. Hold out ~100 Eedi questions where all three distractors are tagged (seeded; never used for any tuning).
2. In each, replace one real distractor with a junk value (e.g. an implausible number derived from the key, checked so it doesn't accidentally match any misconception's answer). Keep the deleted original.
3. Run the audit tool. Measure:
   - **Junk caught:** % of junk options flagged dead
   - **False alarms:** % of real distractors wrongly flagged dead
   - **Recovery:** % of cases where one of the top-3 suggestions matches the deleted original distractor
4. Answer matching: normalize strings/LaTeX first, try a sympy equivalence check, fall back to an LLM judge only when both fail. Log which method decided each match.

**Don't:** build a UI here. Outputs are JSON report cards per question in `results/audit/`.

### ✅ Checkpoint 2
Show: panel recall, junk-swap metrics, and 5 full report cards (question → panel picks → flags → suggestions). Wait for approval.

---

## Phase 3 — GitHub Page

**Goal:** a static site in `site/` that tells the story clearly to Anand and anyone else.

Sections:
1. **The idea in 30 seconds:** distractors, misconceptions, personas — using the `1/2 + 1/3` example.
2. **Can LLMs be confused on purpose?** Leaderboard (lift by model × framing), the slip / wrong-stupid breakdown, the consistency chart, and the best raw examples.
3. **The audit tool** (if built): 3–5 walkthroughs showing a question → panel picks → dead distractor flagged → suggested replacement. Junk-swap results.
4. **Why it matters:** assessment publishers spend human effort writing and QA-ing item banks; this flags weak distractors automatically.
5. **Method & limitations:** math only, Eedi's misconception tags treated as ground truth, LLM personas aren't real students, what wasn't tested.

Rules:
- Static HTML/CSS/JS reading precomputed JSON. No backend, no API calls from the browser.
- Only show question examples allowed by `LICENSE_NOTES.md`.
- Mobile-friendly. Charts readable without hover.

### ✅ Checkpoint 3
Show the page running locally. Atharva reviews copy and charts before it's published.

---

## Out of scope (future work, don't build)
- Subjects other than math; image-based questions
- Personalised distractors per student
- Validating personas against real student response rates
- Live "paste your question" tool with a backend