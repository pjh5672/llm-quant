"""One config -> one measured run.

Both examples/auto_llm.py and examples/scheme_sweep.py go through here, so a sweep row
and a standalone run of the same config can never drift apart.
"""

import functools
import time
from dataclasses import replace

import torch
from transformers import AutoTokenizer

from llmquant.args import ModelArgs
from llmquant.args.quant_config import BF16, WEIGHT_TARGETS, normalize_dtype
from llmquant.datasets import get_eval_ids
from llmquant.entrypoints.evaluate import evaluate_ppl
from llmquant.entrypoints.oneshot import oneshot
from llmquant.utils.model import load_pretrained
from llmquant.utils.size import model_metrics


@functools.lru_cache(maxsize=4)
def eval_ids(dataset: str, model_id: str):
    """Tokenizing the whole eval split takes a few seconds; a sweep reuses it."""
    return get_eval_ids(dataset, AutoTokenizer.from_pretrained(model_id))


def run_one(config) -> dict:
    """Load, quantize, evaluate. The model is rebuilt per call because fake quant is in place."""
    model, _ = load_pretrained(ModelArgs(model_id=config.model, device=config.device))
    recipe = config.to_modifier()
    cost = model_metrics(model, recipe)
    gb = 1024**3
    if recipe is not None:
        oneshot(model, recipe)

    ids = eval_ids(config.metric or "wikitext2", config.model)
    t0 = time.time()
    ppl = evaluate_ppl(model, ids, config.seq_len)
    # With quantize=False nothing is touched, so report every target as bf16 rather than
    # echoing config values that were never applied. That also makes the baseline row the
    # natural all-bf16 reference for llmquant.analysis.
    quant = config.quant if config.quantize else replace(
        config.quant, **{t: BF16 for t in (*WEIGHT_TARGETS, "activation")}
    )
    row = {
        "name": quant.describe() if config.quantize else "bf16",
        "quantize": config.quantize,
        "attn_weight": normalize_dtype(quant.attn_weight),
        "mlp_weight": normalize_dtype(quant.mlp_weight),
        "head_weight": normalize_dtype(quant.head_weight),
        "activation": normalize_dtype(quant.activation),
        "group_size": quant.group_size,
        "ppl": ppl,
        "size_gb": cost["deployed_bytes"] / gb,
        "decode_gb_per_token": cost["decode_bytes_per_token"] / gb,
        "bits_per_element": cost["bits_per_element"],
        "eval_sec": time.time() - t0,
    }

    del model
    torch.cuda.empty_cache()
    return row


@functools.lru_cache(maxsize=2)
def lambada_examples(limit: int):
    from llmquant.datasets import get_lambada_examples

    return tuple(get_lambada_examples(limit))


def run_generation(config, reference=None):
    """Stage 2: the evaluations that need the decode path, so they only run on a shortlist.

    Returns (metrics, continuations). The continuations of the bf16 run become the reference
    every other run is compared against.
    """
    from llmquant.datasets import GENERATION_PROMPTS
    from llmquant.entrypoints.evaluate import (
        evaluate_lambada,
        generation_agreement,
        greedy_continuations,
    )

    model, tokenizer = load_pretrained(ModelArgs(model_id=config.model, device=config.device))
    recipe = config.to_modifier()
    if recipe is not None:
        oneshot(model, recipe)

    t0 = time.time()
    metrics = {
        "lambada_acc": evaluate_lambada(model, tokenizer, list(lambada_examples(config.lambada_limit)))
    }
    continuations = greedy_continuations(
        model, tokenizer, GENERATION_PROMPTS, config.max_new_tokens
    )
    if reference is not None:
        metrics.update(generation_agreement(reference, continuations))
    metrics["generation_sec"] = time.time() - t0

    del model
    torch.cuda.empty_cache()
    return metrics, continuations
