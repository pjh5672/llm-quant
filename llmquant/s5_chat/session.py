"""Phase 5 -- chat on a quantized model.

This is what the earlier phases were for: a packed file goes in, a conversation comes out,
and the bf16 model is never built. It is also the only place the whole stack runs the way it
would be deployed -- weights read as integers by the kernel, the KV cache quantized if the
config says so, and the cache carried across turns rather than rebuilt.

Carrying the cache is the part worth being careful about. Re-encoding the whole history on
every turn is simpler and, on a quantized cache, quietly different: entries quantized once
and kept accumulate no further error, while re-encoding requantizes everything from scratch
each turn. Keeping the cache is both faster and the behaviour a deployment would have.

A conversation that is carried has to be bounded, because nothing else bounds it. The KV
cache grows by one entry per token forever -- 32 KB per token on Llama-3.2-1B -- and the
run ends either at the model's position limit or at an out-of-memory, both of them far
from here and neither of them legible. So the oldest turns are dropped to stay inside
`max_context_tokens`, which defaults to what the model says it can attend over.

Dropping from the front invalidates the cache: its entries are positional, and the prefix
they were computed from is gone. So a trim resets the cache, and that turn pays a full
re-encode. That is the cost of the bound, and it only arrives once the conversation is
long enough to need it.
"""

from dataclasses import dataclass, field

import torch

SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class ChatSession:
    """A conversation with a model, holding its own KV cache."""

    model: object
    tokenizer: object
    max_new_tokens: int = 256
    system_prompt: str | None = SYSTEM_PROMPT
    cache_factory: object = None
    # None -> ask the model. The generation has to fit inside this too, not just the prompt.
    max_context_tokens: int | None = None
    history: list = field(default_factory=list)
    last_reply_ids: list = field(default_factory=list)
    _cache: object = field(default=None, repr=False)

    def reset(self):
        """Drop the conversation and its cache."""
        self.history.clear()
        self.last_reply_ids = []
        self._cache = None
        return self

    def _messages(self):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        return messages + self.history

    def _context_limit(self) -> int | None:
        if self.max_context_tokens is not None:
            return self.max_context_tokens
        config = getattr(self.model, "config", None)
        return getattr(config, "max_position_embeddings", None) if config else None

    def _encode(self):
        return self.tokenizer.apply_chat_template(
            self._messages(), add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        )

    def _fit_to_context(self):
        """Drop whole turns off the front until the next generation fits.

        Returns True if anything was dropped, which means the cache no longer describes
        this conversation.
        """
        limit = self._context_limit()
        inputs = self._encode()
        if limit is None:
            return inputs, False

        dropped = False
        while inputs["input_ids"].shape[1] + self.max_new_tokens > limit:
            # keep at least the turn just asked; there is nothing useful to return without it
            if len(self.history) <= 1:
                raise ValueError(
                    f"this turn needs {inputs['input_ids'].shape[1] + self.max_new_tokens} "
                    f"tokens of context and the limit is {limit}. Shorten the message or "
                    f"lower max_new_tokens."
                )
            del self.history[0]
            dropped = True
            inputs = self._encode()
        return inputs, dropped

    @torch.no_grad()
    def ask(self, prompt: str) -> str:
        """Add a user turn, generate the reply, keep both in the history."""
        self.history.append({"role": "user", "content": prompt})
        device = next(self.model.parameters()).device
        inputs, dropped = self._fit_to_context()
        inputs = inputs.to(device)
        if dropped:
            self._cache = None  # its entries are positional; the prefix they assumed is gone

        if self._cache is None and self.cache_factory is not None:
            self._cache = self.cache_factory()
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            **({"past_key_values": self._cache} if self._cache is not None else {}),
        )
        new_ids = generated[0, inputs["input_ids"].shape[1] :]
        # kept as ids, not just text: comparing two models means comparing the tokens they
        # chose, and re-tokenizing the decoded string does not always give them back
        self.last_reply_ids = new_ids.tolist()
        reply = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply


def load_for_chat(packed_path=None, config=None, device="cuda"):
    """(model, tokenizer, cache_factory) from a packed file, or from a config to quantize now.

    The packed path is the deployed one -- the architecture comes from the recorded model id
    and the weights straight from the file, so no bf16 copy is ever allocated.
    """
    from transformers import AutoTokenizer

    from llmquant.s1_fake import make_cache_factory

    if packed_path is not None:
        from llmquant.s3_pack import load_packed_model, read_header

        meta = read_header(packed_path)["meta"]
        model = load_packed_model(packed_path, device=device)
        tokenizer = AutoTokenizer.from_pretrained(meta["model_id"])
        kv_bits = meta.get("kv_cache_bits")
    elif config is not None:
        from llmquant.core.config import ModelArgs
        from llmquant.core.model import load_pretrained
        from llmquant.core.oneshot import oneshot

        model, tokenizer = load_pretrained(ModelArgs(model_id=config.model, device=device))
        recipe = config.to_modifier(mode=config.mode)
        if recipe is not None:
            oneshot(model, recipe)
        kv_bits = config.quant.kv_cache_bits if config.quantize else None
    else:
        raise ValueError("chat needs either a packed file or a config")

    return model, tokenizer, make_cache_factory(kv_bits)
