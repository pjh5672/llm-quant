from datasets import load_dataset


def get_wikitext2_test_ids(tokenizer):
    # namespaced for the same reason as openbookqa in tasks.py: the bare name is a
    # redirect that a cold huggingface_hub refuses to parse
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
