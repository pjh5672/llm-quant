import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmquant.core.config import ModelArgs


def load_pretrained(args: ModelArgs):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    return model, tokenizer
