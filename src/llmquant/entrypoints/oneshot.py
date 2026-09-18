from llmquant.args import ModelArgs
from llmquant.modifiers.quantization import QuantizationModifier
from llmquant.utils.model import load_pretrained


def oneshot(model, recipe: QuantizationModifier):
    """Apply recipe in place. `model` is a loaded model or a HF model id."""
    tokenizer = None
    if isinstance(model, str):
        model, tokenizer = load_pretrained(ModelArgs(model_id=model))
    recipe.apply(model)
    return (model, tokenizer) if tokenizer is not None else model
