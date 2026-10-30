# License notes: Eedi – Mining Misconceptions in Mathematics

Source: the competition's Rules page, fetched through the Kaggle API on 2026-10-29 after the rules were accepted on our account. This is a plain reading of the rules, **not legal advice**.

## What the rules say

Header of the rules:

> **WINNER MODEL CODE LICENSE TYPE**: MIT
> **DATA ACCESS AND USE AND TRAINED MODEL WEIGHTS**: CC BY-NC 4.0

Section 7.A, *Data Access and Use*:

> You may access and use the Competition Data for **non-commercial purposes only**, including for participating in the Competition and on Kaggle.com forums, and for **academic research and education**. [...] The Competition Data is also subject to [...] CC BY-NC 4.0. To the extent that the terms [...] at the URL conflict with or are inconsistent with these Rules, **these Rules will govern**.

Section 7.B, *Data Security*:

> You agree **not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition.**

Section A.1 (winner license) also says that "training data may not be used for commercial purposes".

**Key point:** CC BY-NC on its own would allow non-commercial redistribution with attribution. However, the rules override CC BY-NC wherever the two conflict, and 7.B forbids publishing or redistributing the data to non-participants. The stricter rule wins.

## Answers to the README's questions

### Can we show example questions publicly on the GitHub Page?

**No, not by default.** Publishing questions, options or misconception names from the dataset counts as "publishing [...] the Competition Data". A handful of short quoted examples in a research write-up *might* be defensible as minimal quotation, but that is a judgement call.

What the page can safely show:
- Aggregate metrics: hit/slip/lift rates, charts, counts.
- Our own **illustrative examples written by us** in Eedi's style, such as the `1/2 + 1/3 → 2/5` example in the README.
- Question IDs, construct and subject names at aggregate level (low risk, but these are still dataset fields).

To show real Eedi questions, ask Eedi for written permission. They are the sponsor, and a non-commercial research showcase is a reasonable request.

### Can we use it in a company demo?

**No.** A demo to clients or prospects is commercial use, which 7.A excludes. Internal use as non-commercial research or education is the most the rules allow. Even internal R&D at a company sits in a grey zone, so check with Legal before relying on it.

For a commercial demo, the options are:
1. a licence from Eedi,
2. our own authored question set, or
3. another dataset with a commercial-friendly licence.

The method itself (persona prompts, panel, junk-swap evaluation) is our code and isn't restricted.

## What this means for the repo

- `data/raw/` and **`data/processed/`** are gitignored. `pairs.jsonl` contains the full question text, so committing it to a public repo would count as redistribution.
- `results/` files should reference `question_id` / `misconception_id` and avoid copying question text where possible. Any results file that does include text must stay out of a public repo.
- The site must read only aggregate JSON plus hand-written examples, unless Eedi gives permission.
- **Open question for Atharva:** each LLM call sends question text to a model provider through LLM Foundry. That is standard practice for inference, but it can be read as "transmitting" data to a third party under 7.B. It's probably fine for non-commercial research, but someone should be aware of it.
