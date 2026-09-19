"""Generation tasks: the model writes tokens and they are matched against a known answer.

Unlike PPL and LAMBADA (both teacher-forced single forward passes), these run the decode
loop and score what actually came out, on an absolute scale. Accuracy across a suite of
them, relative to bf16, is what the selection criterion treats as the accuracy cost.

  arc_easy       ARC-Easy, answered by generating the option letter
  arc_challenge  ARC-Challenge, the harder split of the same set
  openbookqa     OpenBookQA, same shape
  gsm8k          grade-school math with chain of thought, matched on the final number.
                 Far more sensitive to compounding error, but a 1B model scores low
                 enough that noise dominates at any affordable n, so it is opt-in.

Answer parsing is the part that fails silently, so both scorers are pinned by tests.
"""

import re

from datasets import load_dataset

ANSWER_RE = re.compile(r"-?[\d,]*\.?\d+")
CHOICE_INSTRUCTION = "Answer with the letter of the correct option only."
GSM8K_INSTRUCTION = (
    "Think step by step, then give the final answer on its own line as '#### <number>'."
)


def _take(data, limit):
    return data.select(range(min(limit, len(data)))) if limit else data


def _choice_examples(rows, question_key):
    out = []
    for row in rows:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        out.append(
            {
                "prompt": f"{row[question_key]}\n{options}\n\n{CHOICE_INSTRUCTION}",
                "answer": row["answerKey"],
                "choices": list(labels),
            }
        )
    return out


def get_arc(config, limit):
    return _choice_examples(_take(load_dataset("allenai/ai2_arc", config, split="test"), limit), "question")


def get_openbookqa(limit):
    return _choice_examples(_take(load_dataset("openbookqa", "main", split="test"), limit), "question_stem")


def get_gsm8k(limit):
    data = _take(load_dataset("openai/gsm8k", "main", split="test"), limit)
    return [
        {
            "prompt": f"{row['question']}\n\n{GSM8K_INSTRUCTION}",
            "answer": row["answer"].rsplit("####", 1)[-1].strip(),
            "choices": None,
        }
        for row in data
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
    "arc_easy": {
        "loader": lambda limit: get_arc("ARC-Easy", limit),
        "score": _score_choice,
        "max_new_tokens": 8,
        "limit": 500,
    },
    "arc_challenge": {
        "loader": lambda limit: get_arc("ARC-Challenge", limit),
        "score": _score_choice,
        "max_new_tokens": 8,
        "limit": 500,
    },
    "openbookqa": {
        "loader": get_openbookqa,
        "score": _score_choice,
        "max_new_tokens": 8,
        "limit": 500,
    },
    "gsm8k": {
        "loader": get_gsm8k,
        "score": _score_number,
        "max_new_tokens": 256,
        "limit": 100,
    },
}
DEFAULT_TASKS = ("arc_easy", "arc_challenge", "openbookqa")


def get_generation_task(name: str, limit: int | None = None):
    if name not in TASKS:
        raise ValueError(f"unknown generation task {name!r}, expected one of {sorted(TASKS)}")
    spec = TASKS[name]
    return spec["loader"](limit if limit is not None else spec["limit"]), spec
