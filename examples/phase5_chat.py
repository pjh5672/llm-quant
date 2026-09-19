"""Chat with a quantized model.

    python examples/phase5_chat.py --load-packed experiments/w4/model.bin
    python examples/phase5_chat.py --cfg configs/phase1/w8a16.yaml --mode kernel
    python examples/phase5_chat.py --load-packed model.bin --ask "What is a prime number?"

With --load-packed the bf16 model is never built: the architecture comes from the model id
recorded in the file and the weights straight from the packed integers. The KV cache dtype
comes from the file too, since it is applied at generate time and is not part of the
weights -- without it a packed W4 model would quietly chat with a bf16 cache.
"""

import argparse
from pathlib import Path

import torch

from llmquant.core.parser import build_config
from llmquant.stages.s5_chat import ChatSession, load_for_chat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-packed", type=str, default=None, help="a .bin from --pack")
    parser.add_argument("--cfg", type=str, default=None, help="quantize now, from a config")
    parser.add_argument("--mode", type=str, default=None, choices=("fake", "real", "kernel"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--ask", type=str, default=None, help="one question, then exit")
    args = parser.parse_args()

    if not args.load_packed and not args.cfg:
        parser.error("pass --load-packed or --cfg")

    config = None
    if args.cfg:
        argv = ["--cfg", args.cfg] + (["--mode", args.mode] if args.mode else [])
        config = build_config(argv, root_dir=ROOT)

    torch.manual_seed(0)
    model, tokenizer, cache_factory = load_for_chat(
        packed_path=args.load_packed, config=config, device=args.device
    )
    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        cache_factory=cache_factory,
    )
    source = args.load_packed or f"{args.cfg} (mode={config.mode})"
    print(f"loaded {source}; KV cache {'quantized' if cache_factory else 'bf16'}")

    if args.ask:
        print(session.ask(args.ask))
        return

    print("type /reset to clear the conversation, /exit to leave")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt == "/exit":
            break
        if prompt == "/reset":
            session.reset()
            print("(conversation cleared)")
            continue
        print(f"\nmodel> {session.ask(prompt)}")


if __name__ == "__main__":
    main()
