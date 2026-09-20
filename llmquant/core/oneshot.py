from llmquant.core.config import ModelArgs
from llmquant.core import QuantizationModifier
from llmquant.core.model import load_pretrained


def oneshot(model, recipe: QuantizationModifier):
    """Apply recipe in place. `model` is a loaded model or a HF model id."""
    tokenizer = None
    if isinstance(model, str):
        model, tokenizer = load_pretrained(ModelArgs(model_id=model))
    recipe.apply(model)
    return (model, tokenizer) if tokenizer is not None else model
