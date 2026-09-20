import argparse
import time

import torch
import torch.nn.functional as F

from llmquant.core.config import DatasetArgs, ModelArgs
from llmquant.core.datasets import get_eval_ids
from llmquant.core.model import load_pretrained


@torch.no_grad()
def evaluate_ppl(model, input_ids, seqlen=2048):
    device = next(model.parameters()).device
    n_samples = input_ids.numel() // seqlen
    nll_sum, n_tokens = 0.0, 0
    for i in range(n_samples):
        batch = input_ids[:, i * seqlen : (i + 1) * seqlen].to(device)
        logits = model(batch).logits.float()
        nll_sum += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        n_tokens += batch.size(1) - 1
    return torch.exp(torch.tensor(nll_sum / n_tokens)).item()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=64):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(next(model.parameters()).device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="bf16 baseline PPL + generation check")
    parser.add_argument("--model-id", default=ModelArgs.model_id)
    parser.add_argument("--seqlen", type=int, default=DatasetArgs.seqlen)
    parser.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    args = parser.parse_args()

    model, tokenizer = load_pretrained(ModelArgs(model_id=args.model_id))
    ids = get_eval_ids(DatasetArgs.dataset, tokenizer)
    t0 = time.time()
    ppl = evaluate_ppl(model, ids, args.seqlen)
    print(f"wikitext2 test PPL (seqlen={args.seqlen}): {ppl:.4f}  [{time.time() - t0:.1f}s]")
    print(f"generation: {generate(model, tokenizer, args.prompt)!r}")


if __name__ == "__main__":
    main()


@torch.no_grad()
def evaluate_lambada(model, tokenizer, examples, batch_size: int = 8) -> float:
    """Last-word accuracy: the target counts only if *every* one of its tokens is the argmax.

    Teacher-forced, which is the standard LAMBADA definition (lm-eval's `is_greedy`). Unlike
    perplexity one wrong token fails the whole example, so it degrades visibly where an
    averaged log-likelihood barely moves.
    """
    device = next(model.parameters()).device
    correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        for context, target in batch:
            ctx = tokenizer(context, return_tensors="pt").input_ids.to(device)
            tgt = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            ids = torch.cat([ctx, tgt], dim=1)
            logits = model(ids).logits
            # position i predicts token i+1, so the target starts at ctx_len - 1
            predicted = logits[0, ctx.shape[1] - 1 : -1].argmax(dim=-1)
            correct += int(torch.equal(predicted, tgt[0]))
    return correct / len(examples)


@torch.no_grad()
def greedy_continuations(model, tokenizer, prompts, max_new_tokens: int = 64, cache_factory=None):
    """Greedy chat-formatted continuations, as lists of generated token ids.

    This is the only evaluation here that runs the decode loop and the KV cache; PPL and
    LAMBADA are both single forward passes.
    """
    device = next(model.parameters()).device
    out = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            **({"past_key_values": cache_factory()} if cache_factory else {}),
        )
        out.append(generated[0, inputs["input_ids"].shape[1] :].tolist())
    return out


def generation_agreement(reference, candidate) -> dict:
    """How far a quantized model's greedy output tracks the bf16 model's.

    Greedy decoding is deterministic, so any divergence is quantization error. Once the paths
    split they usually stay split, which is why the first divergence point is reported
    alongside the raw token agreement.
    """
    if len(reference) != len(candidate):
        raise ValueError(f"{len(reference)} reference vs {len(candidate)} candidate continuations")
    matched = total = exact = 0
    divergence_fractions = []
    for ref, cand in zip(reference, candidate):
        n = min(len(ref), len(cand))
        agree = [a == b for a, b in zip(ref[:n], cand[:n])]
        matched += sum(agree)
        total += max(len(ref), len(cand))
        first_bad = next((i for i, ok in enumerate(agree) if not ok), None)
        if first_bad is None and len(ref) == len(cand):
            exact += 1
            divergence_fractions.append(1.0)
        else:
            divergence_fractions.append((first_bad if first_bad is not None else n) / max(len(ref), 1))
    return {
        "generation_agreement": matched / total if total else 1.0,
        "generation_exact_match": exact / len(reference),
        "generation_first_divergence": sum(divergence_fractions) / len(divergence_fractions),
    }


@torch.no_grad()
def evaluate_generation_task(
    model, tokenizer, examples, score, max_new_tokens: int, cache_factory=None
) -> float:
    """Accuracy of generated answers against known ones.

    The model writes tokens and they are matched, so this is an absolute score on the real
    decode path -- unlike PPL and LAMBADA, which are teacher-forced single forward passes,
    and unlike generation_agreement, which only measures divergence from bf16.

    Unbatched on purpose: left padding a batch shifts positions and can change greedy
    output, and here the output is the measurement.
    """
    device = next(model.parameters()).device
    correct = 0
    for example in examples:
        messages = [{"role": "user", "content": example["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            **({"past_key_values": cache_factory()} if cache_factory else {}),
        )
        text = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        correct += bool(score(text, example))
    return correct / len(examples)


@torch.no_grad()
def measure_latency(
    model, tokenizer, prompt_tokens: int = 512, new_tokens: int = 64, iters: int = 3,
    cache_factory=None, use_cuda_graph: bool = False,
):
    """Prefill TTFT and steady-state decode throughput, measured separately.

    They live in different regimes and a quantization choice can help one and not the
    other: prefill is compute bound, decode is memory bound and tracks the bytes of weight
    read per token.

    ⚠️ Under mode="fake" these are NOT deployed numbers. Fake quant stores dequantized bf16
    weights, so every combination moves the same bytes and decodes at the same speed, while
    A8 pays an extra activation quant per forward and measures *slower*. Compare against
    decode_bytes_per_token (llmquant.core.metrics) until a real stage exists.
    """
    device = next(model.parameters()).device
    ids = torch.randint(0, tokenizer.vocab_size, (1, prompt_tokens), device=device)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    graphed = False
    if use_cuda_graph:
        from llmquant.stages.s4_kernel import can_graph, graphed_generate

        graphed = can_graph(model, cache_factory)

    def timed(n):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if graphed:
            graphed_generate(model, ids, n)
        else:
            model.generate(
                **inputs,
                max_new_tokens=n,
                min_new_tokens=n,  # keep every iteration the same length
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                **({"past_key_values": cache_factory()} if cache_factory else {}),
            )
        torch.cuda.synchronize()
        return time.perf_counter() - start

    timed(min(4, new_tokens))  # warmup
    torch.cuda.reset_peak_memory_stats(device)
    first = sorted(timed(1) for _ in range(iters))[iters // 2]
    full = sorted(timed(new_tokens) for _ in range(iters))[iters // 2]

    decode_seconds = max(full - first, 1e-9)
    return {
        "ttft_ms": first * 1e3,
        "decode_tps": (new_tokens - 1) / decode_seconds,
        "cuda_graph": graphed,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "latency_prompt_tokens": prompt_tokens,
        "latency_new_tokens": new_tokens,
    }
