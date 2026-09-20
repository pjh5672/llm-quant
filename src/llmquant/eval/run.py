"""One config -> one measured run.

examples/auto_llm.py and examples/phase1_sweep.py both go through here, so a sweep row and
a standalone run of the same config cannot drift apart.

What gets measured, and why:

  tasks    generation tasks (ARC, OpenBookQA, GSM8K). The model writes tokens and they are
           matched against known answers, so this is an absolute score on the real decode
           path. This is the accuracy number the selection criterion uses.
  ppl      wikitext perplexity. Kept as a secondary signal because it is cheap and
           comparable to the literature, but it is teacher-forced and understates the
           damage: a combination reading +0.11% PPL still changed 12.5% of generations.
  latency  TTFT and decode throughput, measured separately because prefill is compute
           bound and decode is memory bound.
  cost     disk bytes, decode bytes per token, bits per element.
"""

import functools
import time
from dataclasses import replace

import torch
from transformers import AutoTokenizer

from llmquant.core.config import BF16, WEIGHT_TARGETS, ModelArgs, normalize_dtype
from llmquant.core.datasets import get_eval_ids, get_generation_task
from llmquant.core.metrics import model_metrics
from llmquant.core.model import load_pretrained
from llmquant.core.oneshot import oneshot
from llmquant.stages.s1_fake import make_cache_factory
from llmquant.eval.evaluate import (
    evaluate_generation_task,
    evaluate_lambada,
    evaluate_ppl,
    generation_agreement,
    greedy_continuations,
    measure_latency,
)

GB = 1024**3


@functools.lru_cache(maxsize=4)
def eval_ids(dataset: str, model_id: str):
    """Tokenizing the whole eval split takes a few seconds; a sweep reuses it."""
    return get_eval_ids(dataset, AutoTokenizer.from_pretrained(model_id))


@functools.lru_cache(maxsize=8)
def task_examples(name: str, limit):
    examples, spec = get_generation_task(name, limit)
    return tuple(examples), spec


@functools.lru_cache(maxsize=2)
def lambada_examples(limit: int):
    from llmquant.core.datasets import get_lambada_examples

    return tuple(get_lambada_examples(limit))


def _applied(config):
    """The dtype fields as actually applied; with quantize=False nothing was."""
    if config.quantize:
        return config.quant
    return replace(config.quant, **{t: BF16 for t in (*WEIGHT_TARGETS, "activation")})


def run_one(config) -> dict:
    """Load, quantize, measure. The model is rebuilt per call because fake quant is in place."""
    if config.load_packed:
        # the whole point of a packed file: the bf16 model is never built, so there is also
        # nothing to cost the quantized one against
        from llmquant.stages.s3_pack import load_packed_model

        model = load_packed_model(config.load_packed, device=config.device)
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        cost = None
    else:
        model, tokenizer = load_pretrained(ModelArgs(model_id=config.model, device=config.device))
        recipe = config.to_modifier()
        cost = model_metrics(model, recipe)
        if recipe is not None:
            oneshot(model, recipe)
        if config.pack:
            from llmquant.stages.s3_pack import save_packed_model

            save_packed_model(
                model,
                config.project_dir / "model.bin",
                config.model,
                kv_cache_bits=config.quant.kv_cache_bits if config.quantize else None,
            )

    quant = _applied(config)
    # the KV cache is quantized at generation time, not by rewiring the model, so it only
    # shows up in the generation tasks -- which is exactly why accuracy moved off PPL,
    # since a PPL pass never reads the cache back
    cache_factory = make_cache_factory(quant.kv_cache_bits if config.quantize else None)
    started = time.time()
    row = {
        "name": quant.describe() if config.quantize else "bf16",
        "quantize": config.quantize,
        "attn_weight": normalize_dtype(quant.attn_weight),
        "mlp_weight": normalize_dtype(quant.mlp_weight),
        "head_weight": normalize_dtype(quant.head_weight),
        "activation": normalize_dtype(quant.activation),
        "kv_cache": normalize_dtype(quant.kv_cache),
        "group_size": quant.group_size,
    }
    # a model loaded from a packed file has no bf16 original to cost against
    if cost is not None:
        row.update(
            kv_gb_per_1k_context=cost["kv_bytes_per_token"] * 1024 / GB,
            size_gb=cost["deployed_bytes"] / GB,
            decode_gb_per_token=cost["decode_bytes_per_token"] / GB,
            bits_per_element=cost["bits_per_element"],
        )

    accuracies = {}
    for name in config.tasks:
        examples, spec = task_examples(name, config.task_limit)
        accuracies[name] = evaluate_generation_task(
            model, tokenizer, list(examples), spec["score"], spec["max_new_tokens"], cache_factory
        )
    if accuracies:
        row["task_acc"] = accuracies
        row["mean_task_acc"] = sum(accuracies.values()) / len(accuracies)

    if config.ppl:
        ids = eval_ids(config.metric or "wikitext2", config.model)
        row["ppl"] = evaluate_ppl(model, ids, config.seq_len)

    if config.latency:
        row.update(
            measure_latency(
                model,
                tokenizer,
                prompt_tokens=config.latency_prompt_tokens,
                new_tokens=config.latency_new_tokens,
                cache_factory=cache_factory,
                use_cuda_graph=config.cuda_graph,
            )
        )
        # which path produced those timings. On the fake path they measure the simulation,
        # not a deployment, and the score has to know that -- see prefill_is_measurable.
        row["latency_mode"] = "packed" if config.load_packed else config.mode

    row["eval_sec"] = time.time() - started
    del model
    torch.cuda.empty_cache()
    return row


def run_generation(config, reference=None):
    """Stage 2: the relative generation check, run only on a shortlist.

    Returns (metrics, continuations). The bf16 run's continuations become the reference
    everything else is compared against.
    """
    from llmquant.core.datasets import GENERATION_PROMPTS

    model, tokenizer = load_pretrained(ModelArgs(model_id=config.model, device=config.device))
    recipe = config.to_modifier()
    if recipe is not None:
        oneshot(model, recipe)
    cache_factory = make_cache_factory(config.quant.kv_cache_bits if config.quantize else None)

    started = time.time()
    metrics = {
        "lambada_acc": evaluate_lambada(
            model, tokenizer, list(lambada_examples(config.lambada_limit))
        )
    }
    continuations = greedy_continuations(
        model, tokenizer, GENERATION_PROMPTS, config.max_new_tokens, cache_factory
    )
    if reference is not None:
        metrics.update(generation_agreement(reference, continuations))
    metrics["generation_sec"] = time.time() - started

    del model
    torch.cuda.empty_cache()
    return metrics, continuations
