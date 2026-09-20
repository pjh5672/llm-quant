"""Replay a decode step instead of relaunching it.

This belongs to Phase 4 because Phase 4 is the deployed path. A quantized decode step is
mostly bandwidth: the weight GEMVs read 1.21 GB per token and the kernel already moves them
at 77% of peak, which leaves little for a better kernel to win. What is left is the cost of
getting those kernels onto the GPU at all -- and a 1B model launches enough small kernels
per token for that to matter. Capturing one step into a CUDA graph and replaying it measured
8.61 -> 5.93 ms, 1.45x, with the logits bit-identical.

It is measured on both the baseline and the quantized runs, never on one alone: graphing
only the quantized side would credit quantization with a speedup that has nothing to do with
it.

Three things this had to learn the hard way, all of them recorded in the docs:

  a StaticCache attends over what it ALLOCATED, not over what is valid, so it must be sized
  to the run and not generously -- a 4096-long cache holding 512 real tokens made the step
  12.64 ms instead of 8.61 ms.

  a StaticCache advances its own write position on every call, so a benchmark loop walks
  off the end of a cache sized for the generation it imitates.

  a replay mutates the cache it was captured against, so correctness cannot be checked by
  rewinding one cache; it takes two, advancing side by side.

A quantized KV cache cannot be captured (it is not a StaticCache), and neither can a model
off CUDA. `graphed_generate` returns None in both cases rather than pretending, and callers
fall back to `model.generate`.
"""

import torch

__all__ = ["GraphedDecoder", "can_graph", "graphed_generate"]

# Forwards run before the capture, to settle the allocator. Each one writes to the cache,
# which is why they are counted into max_cache_len and rewound afterwards.
WARMUP_CALLS = 3


def can_graph(model, cache_factory=None) -> bool:
    """Whether a graphed decode is possible at all for this model and cache choice."""
    if cache_factory is not None:
        return False  # a quantized KV cache is not a StaticCache
    if not torch.cuda.is_available():
        return False
    try:
        return next(model.parameters()).device.type == "cuda"
    except StopIteration:
        return False


class GraphedDecoder:
    """One captured decode step, replayed for every token after the prefill.

    The prefill runs eagerly -- it is one call over many tokens, so there is nothing to
    amortize, and its shape changes with the prompt anyway.
    """

    def __init__(self, model, prompt_tokens: int, new_tokens: int):
        from transformers import StaticCache

        self.model = model
        self.device = next(model.parameters()).device
        # Attention costs the ALLOCATED length, so this is sized to the run. The +
        # WARMUP_CALLS + 1 is not slack: a StaticLayer ignores the cache_position it is
        # handed and writes at its own cumulative_length, advancing it once per call
        # (transformers 5.x), so the warmup and capture calls consume slots too. They are
        # given back by _rewind before any real token is decoded.
        self.max_len = prompt_tokens + new_tokens + WARMUP_CALLS + 1
        self.cache = StaticCache(
            config=model.config, max_batch_size=1, max_cache_len=self.max_len,
            device=self.device, dtype=torch.bfloat16,
        )
        self.token = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.position = torch.zeros(1, dtype=torch.long, device=self.device)
        self.graph = None
        self.logits = None

    def _step(self):
        return self.model(
            self.token, past_key_values=self.cache, use_cache=True,
            cache_position=self.position,
        )

    @torch.no_grad()
    def prefill(self, input_ids):
        out = self.model(
            input_ids, past_key_values=self.cache, use_cache=True,
            cache_position=torch.arange(input_ids.shape[1], device=self.device),
        )
        return out.logits[:, -1:].argmax(-1)

    def _rewind(self, to: int):
        """Put every layer's write cursor back.

        Warming and capturing both run real forwards, so they write real entries at the
        positions the first decoded tokens should occupy. Rewinding makes the next replay
        overwrite them. cumulative_length is a tensor and is mutated in place, which is also
        why a captured graph advances it correctly on replay.
        """
        for layer in self.cache.layers:
            cursor = getattr(layer, "cumulative_length", None)
            if cursor is not None:
                cursor.fill_(to)

    @torch.no_grad()
    def capture(self, position: int):
        """Warm on a side stream, then capture one step. Both are required."""
        self.position.fill_(position)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(WARMUP_CALLS):
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            out = self._step()
        self.logits = out.logits
        # hand back every slot the warmup and the capture consumed
        self._rewind(position)

    @torch.no_grad()
    def step(self, token, position: int):
        """One decoded token. The inputs are written in place so the replay reads them."""
        self.token.copy_(token.reshape(1, 1))
        self.position.fill_(position)
        self.graph.replay()
        return self.logits[:, -1:].argmax(-1)


@torch.no_grad()
def graphed_generate(model, input_ids, max_new_tokens: int):
    """Greedy generation with the decode step replayed. None if it cannot be graphed.

    Returns the full sequence, matching `model.generate(..., do_sample=False)` in shape.
    """
    if not can_graph(model):
        return None
    prompt = input_ids.shape[1]
    decoder = GraphedDecoder(model, prompt, max_new_tokens)
    nxt = decoder.prefill(input_ids)
    out = [nxt]
    if max_new_tokens <= 1:
        return torch.cat([input_ids, nxt], dim=1)

    # capture at the first decode position, then replay for the rest
    decoder.token.copy_(nxt.reshape(1, 1))
    decoder.capture(prompt)
    nxt = decoder.step(nxt, prompt)
    out.append(nxt)
    for i in range(2, max_new_tokens):
        nxt = decoder.step(nxt, prompt + i - 1)
        out.append(nxt)
    return torch.cat([input_ids, *out], dim=1)
