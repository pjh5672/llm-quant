"""Fixed prompts for the generation check.

Hardcoded on purpose: the comparison is against this same model in bf16, so the prompts only
have to be deterministic, offline, and varied enough that a degraded model diverges.
"""

GENERATION_PROMPTS = (
    "What is the capital of France? Answer in one sentence.",
    "Explain what a prime number is to a ten year old.",
    "Write a haiku about winter mornings.",
    "List three uses for baking soda.",
    "Summarize why the sky appears blue.",
    "What is the difference between RAM and a hard drive?",
    "Give me a simple recipe for tomato soup.",
    "Translate 'good morning, how are you?' into Spanish.",
    "What year did the first moon landing happen, and who was involved?",
    "Describe the water cycle in three steps.",
    "Write a short function in Python that reverses a string.",
    "What are the main causes of inflation?",
    "Name four planets in the solar system and one fact about each.",
    "How do vaccines work?",
    "Give three tips for sleeping better.",
    "What is the plot of Romeo and Juliet in two sentences.",
)
