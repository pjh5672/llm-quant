from datasets import load_dataset


def get_lambada_examples(limit: int | None = 1000):
    """(context, target) pairs; the task is to produce the final word of each passage.

    Accuracy here is a *task* number rather than a perplexity, and unlike wikitext PPL a
    single wrong token costs a whole example, which makes it sensitive to quantization.
    """
    data = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    texts = data["text"][:limit] if limit else data["text"]
    out = []
    for text in texts:
        context, _, last_word = text.strip().rpartition(" ")
        if context and last_word:
            out.append((context, " " + last_word))
    return out
