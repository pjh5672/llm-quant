"""Phase 5 -- chat on a quantized model.

This is what the earlier phases were for: a packed file goes in, a conversation comes out,
and the bf16 model is never built. It is also the only place the whole stack runs the way it
would be deployed -- weights read as integers by the kernel, the KV cache quantized if the
config says so, and the cache carried across turns rather than rebuilt.

Carrying the cache is the part worth being careful about. Re-encoding the whole history on
every turn is simpler and, on a quantized cache, quietly different: entries quantized once
and kept accumulate no further error, while re-encoding requantizes everything from scratch
each turn. Keeping the cache is both faster and the behaviour a deployment would have.
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
    history: list = field(default_factory=list)
    _cache: object = field(default=None, repr=False)

    def reset(self):
        """Drop the conversation and its cache."""
        self.history.clear()
        self._cache = None
        return self

    def _messages(self):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        return messages + self.history

    @torch.no_grad()
    def ask(self, prompt: str) -> str:
        """Add a user turn, generate the reply, keep both in the history."""
        self.history.append({"role": "user", "content": prompt})
        device = next(self.model.parameters()).device
        inputs = self.tokenizer.apply_chat_template(
            self._messages(), add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)

        if self._cache is None and self.cache_factory is not None:
            self._cache = self.cache_factory()
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            **({"past_key_values": self._cache} if self._cache is not None else {}),
        )
        reply = self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply


def load_for_chat(packed_path=None, config=None, device="cuda"):
    """(model, tokenizer, cache_factory) from a packed file, or from a config to quantize now.

    The packed path is the deployed one -- the architecture comes from the recorded model id
    and the weights straight from the file, so no bf16 copy is ever allocated.
    """
    from transformers import AutoTokenizer

    from llmquant.stages.s1_fake import make_cache_factory

    if packed_path is not None:
        from llmquant.stages.s3_pack import load_packed_model, read_header

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
