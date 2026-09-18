from llmquant.datasets.lambada import get_lambada_examples
from llmquant.datasets.prompts import GENERATION_PROMPTS
from llmquant.datasets.wikitext import get_wikitext2_test_ids

EVAL_DATASETS = {
    "wikitext2": get_wikitext2_test_ids,
}


def get_eval_ids(name, tokenizer):
    try:
        loader = EVAL_DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}, expected one of {list(EVAL_DATASETS)}")
    return loader(tokenizer)


__all__ = [
    "EVAL_DATASETS",
    "GENERATION_PROMPTS",
    "get_eval_ids",
    "get_lambada_examples",
]
