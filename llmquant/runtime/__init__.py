"""Running a quantized model: generating from it, and talking to it.

The decode loop and the conversation around it, which is where the weights stop being a
measurement and start being something someone uses. graph_decode replays a captured decode
step rather than relaunching it; chat carries the conversation and its KV cache across
turns; compare puts the quantized model beside the bf16 one and reports where they part.
"""

from llmquant.runtime.chat import SYSTEM_PROMPT, ChatSession, load_for_chat
from llmquant.runtime.compare import (
    ComparisonSession,
    ComparisonTurn,
    load_for_comparison,
)
from llmquant.runtime.graph_decode import (
    WARMUP_CALLS,
    GraphedDecoder,
    can_graph,
    graphed_generate,
)

__all__ = [
    "ChatSession",
    "ComparisonSession",
    "ComparisonTurn",
    "GraphedDecoder",
    "SYSTEM_PROMPT",
    "WARMUP_CALLS",
    "can_graph",
    "graphed_generate",
    "load_for_chat",
    "load_for_comparison",
]
