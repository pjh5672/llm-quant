"""Chatting with both models at once, and reporting where they part.

The stub covers the bookkeeping -- which model answers, what gets recorded, how divergence
is scored -- because that is ordinary logic and does not need a GPU to be wrong. The one
thing that does need the real stack is that a quantized model and a bf16 one actually
produce comparable text, and that is a single marked test.
"""

import pytest
import torch

from llmquant.runtime import ChatSession, ComparisonSession


class Encoding(dict):
    def to(self, device):
        return self


class StubTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return Encoding(input_ids=torch.zeros(1, len(messages), dtype=torch.long))

    def decode(self, ids, **kwargs):
        return " ".join(str(int(i)) for i in ids)


class ScriptedModel:
    """Emits a fixed sequence of token ids, so divergence is exactly known."""

    def __init__(self, tokens):
        self.tokens = tokens

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, input_ids=None, **kwargs):
        reply = torch.tensor([self.tokens], dtype=torch.long)
        return torch.cat([input_ids, reply], dim=1)


def _session(reference_tokens, candidate_tokens, **kw):
    return ComparisonSession(
        candidate=ChatSession(model=ScriptedModel(candidate_tokens),
                              tokenizer=StubTokenizer(), system_prompt=None),
        reference=ChatSession(model=ScriptedModel(reference_tokens),
                              tokenizer=StubTokenizer(), system_prompt=None),
        **kw,
    )


def test_identical_replies_are_reported_as_identical():
    turn = _session([1, 2, 3], [1, 2, 3]).ask("hello")
    assert turn.exact
    assert turn.agreement == 1.0
    assert turn.first_divergence is None
    assert "identical" in turn.format()


def test_the_first_differing_token_is_located():
    turn = _session([1, 2, 3, 4], [1, 2, 9, 4]).ask("hello")
    assert not turn.exact
    assert turn.first_divergence == 2
    assert "first difference at token 2" in turn.format()


def test_a_reply_that_stops_early_still_counts_as_divergence():
    turn = _session([1, 2, 3, 4], [1, 2]).ask("hello")
    assert not turn.exact
    assert turn.first_divergence == 2
    assert turn.reference_tokens == 4 and turn.candidate_tokens == 2


def test_both_models_keep_their_own_history_by_default():
    """Divergence compounds, which is what someone using it would experience."""
    session = _session([1, 2], [3, 4])
    session.ask("first")
    session.ask("second")
    assert session.candidate.history[1]["content"] != session.reference.history[1]["content"]


def test_follow_reference_puts_both_on_the_same_footing():
    """Each turn is then judged on its own rather than on the drift before it."""
    session = _session([1, 2], [3, 4], follow_reference=True)
    session.ask("first")
    assert session.candidate.history[1] == session.reference.history[1]
    assert session.candidate._cache is None  # the cache described the reply it replaced


def test_summary_covers_the_conversation():
    session = _session([1, 2], [1, 2])
    session.ask("a")
    session.candidate.model.tokens = [9, 9]
    session.ask("b")
    summary = session.summary()
    assert summary["turns"] == 2
    assert summary["identical_replies"] == 1
    assert 0.0 < summary["mean_agreement"] < 1.0


def test_reset_clears_both_sides():
    session = _session([1], [1])
    session.ask("hello")
    session.reset()
    assert session.turns == [] and session.candidate.history == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_quantized_model_can_be_compared_against_bf16():
    from pathlib import Path

    from llmquant.core.parser import build_config
    from llmquant.runtime import load_for_comparison

    config = build_config(
        ["--cfg", "configs/recommended.yaml", "--mode", "kernel"], root_dir=Path(".")
    )
    session = load_for_comparison(config=config, max_new_tokens=12)
    turn = session.ask("What is the capital of France?")
    assert turn.reference and turn.candidate
    assert 0.0 <= turn.agreement <= 1.0
    # W8 on this model reproduces bf16 on a question this easy
    assert "paris" in turn.candidate.lower()


# ---------------------------------------------------------------- side-by-side layout

def _turn(reference, candidate, **kw):
    from llmquant.runtime import ComparisonTurn

    base = {"prompt": "q", "reference": reference, "candidate": candidate,
            "agreement": 0.5, "exact": reference == candidate, "first_divergence": None,
            "reference_tokens": 1, "candidate_tokens": 1}
    base.update(kw)
    return ComparisonTurn(**base)


def test_the_two_replies_print_in_columns_with_a_rule_between_them():
    text = _turn("alpha beta gamma delta", "alpha beta epsilon zeta").format(width=60)
    lines = text.splitlines()
    assert lines[0].startswith("bf16") and "quantized" in lines[0]
    body = [ln for ln in lines if "|" in ln and "-+-" not in ln]
    assert body, "no column rows"
    for line in body:
        assert line.count("|") >= 1


def test_columns_stay_aligned_when_one_side_is_longer():
    short, long = "yes", " ".join(["word"] * 40)
    text = _turn(long, short).format(width=60)
    rows = [ln for ln in text.splitlines() if "|" in ln and "-+-" not in ln][1:]
    separators = {ln.index("|") for ln in rows}
    assert len(separators) == 1, f"the rule wanders: {separators}"


def test_identical_replies_say_so_under_the_columns():
    assert _turn("same text", "same text").format(width=60).splitlines()[-1] == "identical"


def test_the_verdict_names_where_they_parted():
    turn = _turn("a b c", "a b d", agreement=0.66, exact=False, first_divergence=2)
    assert turn.verdict() == "66% of tokens agree, first difference at token 2"


def test_width_zero_asks_the_terminal():
    """chat.py passes 0 so the layout follows the window rather than a guess."""
    text = _turn("alpha", "beta").format(width=0)
    assert "|" in text and text.splitlines()[-1]
