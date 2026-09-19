"""Phase 5 -- chat (not implemented yet).

Will hold the chat entrypoint that runs on a loaded packed model. This is also where a
generation-based metric would go if kv_cache quantization is ever implemented: the PPL
used in Phase 1 never reads the KV cache, so it cannot see that change at all.
"""
