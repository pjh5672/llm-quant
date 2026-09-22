"""Phase 5: the conversation layer.

The history and cache bookkeeping is tested against a stub, because it is ordinary logic
and does not need a GPU to be wrong. The one thing that does need the real stack -- a packed
file producing a coherent multi-turn exchange -- is a single marked test.
"""

import pytest
import torch

from llmquant.runtime import SYSTEM_PROMPT, ChatSession


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
    from llmquant.packing import save_packed_model
    from llmquant.runtime import load_for_chat

    model, _ = load_pretrained(ModelArgs())
    config = QuantConfig(attn_weight="int8", mlp_weight="int8")
    oneshot(model, config.to_modifier(mode="kernel"))
    path = save_packed_model(
        model, tmp_path / "chat.bin", ModelArgs().model_id, kv_cache_bits=config.kv_cache_bits
    )
    del model
    torch.cuda.empty_cache()

    loaded, tokenizer, _cache_factory = load_for_chat(packed_path=path)
    chat = ChatSession(model=loaded, tokenizer=tokenizer, max_new_tokens=24)
    first = chat.ask("Name the capital of France in one word.")
    assert "paris" in first.lower()
    chat.ask("And of Japan?")
    assert len(chat.history) == 4


class CountingTokenizer(StubTokenizer):
    """Encodes one token per message, so a context limit can be reached in a few turns."""

    def apply_chat_template(self, messages, **kwargs):
        self.seen.append([dict(m) for m in messages])
        return Encoding(input_ids=torch.zeros(1, len(messages), dtype=torch.long))


class RecordingModel(StubModel):
    """Records the prompt length each turn actually generated from.

    `tokenizer.seen` cannot answer that: trimming re-encodes, so it also holds the
    over-limit attempts that were rejected on the way down.
    """

    def __init__(self):
        super().__init__()
        self.prompt_lengths = []

    def generate(self, input_ids=None, **kwargs):
        self.prompt_lengths.append(input_ids.shape[1])
        return super().generate(input_ids=input_ids, **kwargs)


def test_old_turns_are_dropped_to_stay_inside_the_context():
    """Nothing else bounds a carried conversation: the cache grows one entry per token
    until the model's position limit or an out-of-memory, neither of them legible."""
    session = ChatSession(model=RecordingModel(), tokenizer=CountingTokenizer(),
                          max_new_tokens=1, max_context_tokens=4, system_prompt=None)
    for i in range(6):
        session.ask(f"turn {i}")
    # every turn generated from a prompt that left room for the generation
    assert all(n + 1 <= 4 for n in session.model.prompt_lengths), session.model.prompt_lengths
    assert session.history[-2]["content"] == "turn 5"   # the newest turn is never dropped
    assert all(m["content"] != "turn 0" for m in session.history)  # the oldest is gone


def test_trimming_drops_the_cache_because_its_entries_are_positional():
    built = []

    def factory():
        cache = object()
        built.append(cache)
        return cache

    session = ChatSession(model=StubModel(), tokenizer=CountingTokenizer(),
                          max_new_tokens=1, max_context_tokens=4, system_prompt=None,
                          cache_factory=factory)
    for i in range(6):
        session.ask(f"turn {i}")
    assert len(built) > 1, "a trim must invalidate the cache it no longer describes"


def test_a_single_turn_that_cannot_fit_says_so():
    session = ChatSession(model=StubModel(), tokenizer=CountingTokenizer(),
                          max_new_tokens=99, max_context_tokens=2, system_prompt=None)
    with pytest.raises(ValueError, match="tokens of context and the limit is 2"):
        session.ask("hello")


def test_without_a_limit_nothing_is_dropped():
    class NoLimitModel(StubModel):
        config = None

    session = ChatSession(model=NoLimitModel(), tokenizer=CountingTokenizer(),
                          max_new_tokens=1, system_prompt=None)
    for i in range(6):
        session.ask(f"turn {i}")
    assert len(session.history) == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_load_for_chat_accepts_a_config():
    """It passed mode= to RunConfig.to_modifier(), which takes no arguments -- so chatting
    from a config rather than a packed file raised every time."""
    from pathlib import Path

    from llmquant.core.parser import build_config
    from llmquant.runtime import load_for_chat

    config = build_config(
        ["--cfg", "configs/recommended.yaml", "--mode", "fake"], root_dir=Path(".")
    )
    model, tokenizer, _cache_factory = load_for_chat(config=config)
    assert model is not None and tokenizer is not None


def test_streaming_is_off_unless_asked_for():
    session = ChatSession(model=StubModel(), tokenizer=StubTokenizer())
    assert session.stream is False


def test_a_streaming_session_hands_generate_a_streamer():
    session = ChatSession(model=StubModel(), tokenizer=StubTokenizer(), stream=True)
    session.ask("hello")
    assert "streamer" in session.model.calls[0]


def test_a_quiet_session_does_not():
    session = ChatSession(model=StubModel(), tokenizer=StubTokenizer())
    session.ask("hello")
    assert "streamer" not in session.model.calls[0]


def test_the_default_ceiling_leaves_room_for_long_answers():
    """256 was a silent cap on anything essay-length; the guard reserves whatever this is
    out of the context budget, so it has to be a deliberate number."""
    assert ChatSession(model=StubModel(), tokenizer=StubTokenizer()).max_new_tokens >= 2048
