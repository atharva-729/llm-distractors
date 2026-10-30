"""Persona prompt templates. Freeze after the pilot: no tuning on the full run."""
import re

LETTERS = ["A", "B", "C", "D"]

SINGLE = {
    "student": ("You are a student who is normal in every way except you firmly believe: "
                "{misconception}. Answer the question as this student would. "
                "Reply with one letter only."),
    "teacher": ("A student holds this misconception: {misconception}. "
                "Predict which option this student would pick. Reply with one letter only."),
    "none": "Answer the question. Reply with one letter only.",
}

# Batched: several independent items in one call. Each item carries its own persona.
BATCHED = {
    "student": ("You will see {n} independent items. Each item states a belief and gives a "
                "multiple-choice question. For each item, you are a different student who is "
                "normal in every way except you firmly believe that item's belief. "
                "Answer each question as that student would."),
    "teacher": ("You will see {n} independent items. Each item describes a student's "
                "misconception and gives a multiple-choice question. For each item, predict "
                "which option that student would pick."),
    "none": "You will see {n} independent multiple-choice questions. Answer each one.",
}
BATCHED_FORMAT = (" Reply with exactly one line per item in the form `<item number>: <letter>`, "
                  "and nothing else.")
BELIEF_LABEL = {"student": "Belief", "teacher": "Student's misconception"}

RETRY_SINGLE = "Reply with exactly one letter: A, B, C or D."
RETRY_BATCHED = "Reply with exactly one line per item in the form `<item number>: <letter>`."


def question_block(question_text, shown_options):
    """shown_options: list of option texts in display order (A, B, C, D)."""
    opts = "\n".join(f"{L}) {t}" for L, t in zip(LETTERS, shown_options))
    return f"Question:\n{question_text}\n\nOptions:\n{opts}"


# Pilot 2: Claude and Gemini ignored the system-only format instruction and began
# explaining, so it is repeated at the end of the user turn for every model, and
# single calls use structured output constrained to one letter (same for all models:
# the answer method itself shifted picks in testing, so it must not vary by model).
ANSWER_SUFFIX = "\n\nReply with one letter only (A, B, C or D)."
ANSWER_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "answer", "strict": True,
    "schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": LETTERS}},
               "required": ["answer"], "additionalProperties": False}}}


def single_messages(framing, misconception, question_text, shown_options):
    system = SINGLE[framing].format(misconception=misconception)
    return [{"role": "system", "content": system},
            {"role": "user", "content": question_block(question_text, shown_options) + ANSWER_SUFFIX}]


def batched_messages(framing, items):
    """items: list of (misconception or None, question_text, shown_options)."""
    system = BATCHED[framing].format(n=len(items)) + BATCHED_FORMAT
    parts = []
    for i, (mc, qt, opts) in enumerate(items, 1):
        head = f"Item {i}"
        if framing != "none":
            head += f"\n{BELIEF_LABEL[framing]}: {mc}"
        parts.append(f"{head}\n{question_block(qt, opts)}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n---\n\n".join(parts)}]


_SINGLE_RE = re.compile(r"^\W*([ABCD])(?:\W|$)")
_LINE_RE = re.compile(r"^\W*(?:item\s*)?(\d+)\s*[:.)\-]\s*\**\(?([ABCD])\b", re.IGNORECASE | re.MULTILINE)


_ANSWER_RE = re.compile(r'"answer"\s*:\s*"([ABCD])')


def parse_single(text):
    """Structured reply {"answer": "B"}, falling back to a bare leading letter.
    Regex rather than json.loads: Gemini sometimes stops right after the letter."""
    m = _ANSWER_RE.search(text) or _SINGLE_RE.match(text.strip())
    return m.group(1) if m else None


def parse_batched(text, n):
    """Return {item_number: letter}; items missing or out of range are absent."""
    out = {}
    for num, letter in _LINE_RE.findall(text):
        i = int(num)
        if 1 <= i <= n and i not in out:
            out[i] = letter.upper()
    return out
