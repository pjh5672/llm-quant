"""Generation tasks: the model writes tokens and they are matched against a known answer.

Unlike PPL and LAMBADA (both teacher-forced single forward passes), these run the decode
loop and score what actually came out, on an absolute scale.

  arc_easy  multiple choice answered by generating the option letter. Short outputs, so
            it is cheap and n can be large enough for a tight confidence interval.
  gsm8k     grade-school math with chain of thought, matched on the final number. Long
            outputs make it far more sensitive to compounding error, but a 1B model scores
            low enough that the noise dominates at any affordable n.
"""

import re

from datasets import load_dataset

ANSWER_RE = re.compile(r"-?[\d,]*\.?\d+")


def get_arc_easy(limit: int | None = 500):
    data = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    rows = data.select(range(min(limit, len(data)))) if limit else data
    out = []
    for row in rows:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        out.append(
            {
                "prompt": f"{row['question']}\n{options}\n\n"
                "Answer with the letter of the correct option only.",
                "answer": row["answerKey"],
                "choices": list(labels),
            }
        )
    return out


def get_gsm8k(limit: int | None = 100):
    data = load_dataset("openai/gsm8k", "main", split="test")
    rows = data.select(range(min(limit, len(data)))) if limit else data
    return [
        {
            "prompt": f"{row['question']}\n\nThink step by step, then give the final "
            "answer on its own line as '#### <number>'.",
            "answer": row["answer"].rsplit("####", 1)[-1].strip(),
            "choices": None,
        }
        for row in rows
    ]


def _score_choice(text, example):
    """First standalone option label in the output; models like to pad with prose.

    Case-sensitive first: in "the answer is a question about A" the English article would
    otherwise be read as option A. Only if no exact-case label appears do we fall back,
    so a model that answers in lowercase still scores.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    for match_case in (True, False):
        for token in tokens:
            candidate = token if match_case else token.upper()
            if candidate in example["choices"]:
                return candidate == example["answer"].upper()
    return False


def _score_number(text, example):
    """The number after the last '####', else the last number anywhere in the output."""
    tail = text.rsplit("####", 1)[-1] if "####" in text else text
    found = ANSWER_RE.findall(tail) or ANSWER_RE.findall(text)
    if not found:
        return False
    return found[-1].replace(",", "").rstrip(".") == example["answer"].replace(",", "")


TASKS = {
    "arc_easy": {"loader": get_arc_easy, "score": _score_choice, "max_new_tokens": 8, "limit": 500},
    "gsm8k": {"loader": get_gsm8k, "score": _score_number, "max_new_tokens": 256, "limit": 100},
}


def get_generation_task(name: str, limit: int | None = None):
    if name not in TASKS:
        raise ValueError(f"unknown generation task {name!r}, expected one of {sorted(TASKS)}")
    spec = TASKS[name]
    return spec["loader"](limit if limit is not None else spec["limit"]), spec
