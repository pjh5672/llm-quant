from dataclasses import dataclass

from llmquant.args.quant_config import BF16, DTYPE_BITS, QuantConfig


@dataclass
class ModelArgs:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass
class DatasetArgs:
    dataset: str = "wikitext2"
    seqlen: int = 2048


__all__ = ["BF16", "DTYPE_BITS", "DatasetArgs", "ModelArgs", "QuantConfig"]
