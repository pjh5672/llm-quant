from datasets import load_dataset


def get_wikitext2_test_ids(tokenizer):
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
