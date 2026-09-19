"""Phase 5 -- chat on a quantized model.

The only place the whole stack runs the way it would be deployed: a packed file in, weights
read as integers by the kernel, the KV cache quantized if the config says so, and the cache
carried across turns instead of rebuilt.
"""

from llmquant.stages.s5_chat.session import SYSTEM_PROMPT, ChatSession, load_for_chat

__all__ = ["SYSTEM_PROMPT", "ChatSession", "load_for_chat"]
