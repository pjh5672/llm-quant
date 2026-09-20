"""Phase 5: the conversation layer.

The history and cache bookkeeping is tested against a stub, because it is ordinary logic
and does not need a GPU to be wrong. The one thing that does need the real stack -- a packed
file producing a coherent multi-turn exchange -- is a single marked test.
"""

import pytest
import torch

from llmquant.s5_chat import SYSTEM_PROMPT, ChatSession


class Encoding(dict):
    """What apply_chat_template returns: a mapping that also knows how to move device."""

    def to(self, device):
        return self


class StubTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, **kwargs):
        self.seen.append([dict(m) for m in messages])
        return Encoding(input_ids=torch.zeros(1, len(messages), dtype=torch.long))

    def decode(self, ids, **kwargs):
        return "  a reply  "


class StubModel:
    def __init__(self):
        self.calls = []

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, input_ids=None, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1, input_ids.shape[1] + 2, dtype=torch.long)


@pytest.fixture
def session():
    return ChatSession(model=StubModel(), tokenizer=StubTokenizer())


def test_a_turn_records_both_sides_and_strips_the_reply(session):
    assert session.ask("hello") == "a reply"
    assert session.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "a reply"},
    ]


def test_the_system_prompt_leads_every_turn_without_entering_the_history(session):
    session.ask("first")
    session.ask("second")
    for messages in session.tokenizer.seen:
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert all(m["role"] != "system" for m in session.history)


def test_history_grows_across_turns(session):
    session.ask("first")
    session.ask("second")
    # the second turn is asked with the first exchange still in front of it
    assert [m["content"] for m in session.tokenizer.seen[-1]][1:] == [
        "first",
        "a reply",
        "second",
    ]


def test_reset_clears_the_conversation_and_the_cache(session):
    session.ask("hello")
    session._cache = object()
    session.reset()
    assert session.history == [] and session._cache is None


def test_the_system_prompt_can_be_turned_off():
    session = ChatSession(model=StubModel(), tokenizer=StubTokenizer(), system_prompt=None)
    session.ask("hello")
    assert session.tokenizer.seen[0][0]["role"] == "user"


def test_the_cache_is_built_once_and_carried(session):
    """Rebuilding it per turn would requantize the whole history every time, which is both
    slower and not what a deployment does."""
    built = []

    def factory():
        cache = object()
        built.append(cache)
        return cache

    session.cache_factory = factory
    session.ask("first")
    session.ask("second")
    assert len(built) == 1
    assert all(call["past_key_values"] is built[0] for call in session.model.calls)


def test_no_cache_is_passed_when_none_is_asked_for(session):
    session.ask("hello")
    assert "past_key_values" not in session.model.calls[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_packed_model_holds_a_two_turn_conversation(tmp_path):

    from llmquant.core.config import ModelArgs, QuantConfig
    from llmquant.core.model import load_pretrained
    from llmquant.core.oneshot import oneshot
    from llmquant.s3_pack import save_packed_model
    from llmquant.s5_chat import load_for_chat

    model, _ = load_pretrained(ModelArgs())
    config = QuantConfig(attn_weight="int8", mlp_weight="int8")
    oneshot(model, config.to_modifier(mode="kernel"))
    path = save_packed_model(
        model, tmp_path / "chat.bin", ModelArgs().model_id, kv_cache_bits=config.kv_cache_bits
    )
    del model
    torch.cuda.empty_cache()

    loaded, tokenizer, cache_factory = load_for_chat(packed_path=path)
    chat = ChatSession(model=loaded, tokenizer=tokenizer, max_new_tokens=24)
    first = chat.ask("Name the capital of France in one word.")
    assert "paris" in first.lower()
    chat.ask("And of Japan?")
    assert len(chat.history) == 4
