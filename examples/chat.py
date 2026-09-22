"""Chat with a quantized model.

    python examples/chat.py --load-packed experiments/w4/model.bin
    python examples/chat.py --cfg configs/w8a16.yaml --mode kernel
    python examples/chat.py --load-packed model.bin --ask "What is a prime number?"
    python examples/chat.py --load-packed model.bin --compare
    python examples/chat.py --load-packed big.bin --compare --reference-device cpu
    python examples/chat.py --load-packed model.bin --max-new-tokens 8192
    python examples/chat.py --load-packed model.bin --no-stream

Replies stream to the terminal as they are decoded, because at 90 tok/s a long answer is
otherwise a minute of silence. `--compare` turns that off: its two columns cannot be filled
token by token at once, so it prints once both models have answered.

With --load-packed the bf16 model is never built: the architecture comes from the model id
recorded in the file and the weights straight from the packed integers. The KV cache dtype
comes from the file too, since it is applied at generate time and is not part of the
weights -- without it a packed W4 model would quietly chat with a bf16 cache.
"""

import argparse
from pathlib import Path

import torch

from llmquant.core.parser import build_config
from llmquant.runtime import ChatSession, load_for_chat, load_for_comparison

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-packed", type=str, default=None, help="a .bin from --pack")
    parser.add_argument("--cfg", type=str, default=None, help="quantize now, from a config")
    parser.add_argument("--mode", type=str, default=None, choices=("fake", "real", "kernel"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="reserved out of the context budget; 8192 works on a model "
                             "whose position limit leaves room for it")
    parser.add_argument("--no-stream", action="store_true",
                        help="wait for the whole reply instead of printing it as it comes")
    parser.add_argument("--ask", type=str, default=None, help="one question, then exit")
    parser.add_argument("--compare", action="store_true",
                        help="answer with the bf16 model too, and report where they part")
    parser.add_argument("--reference-device", type=str, default=None,
                        help="where the bf16 reference goes; use cpu when both will not fit")
    parser.add_argument("--follow-reference", action="store_true",
                        help="feed both models the bf16 reply, so each turn is judged alone")
    args = parser.parse_args()

    if not args.load_packed and not args.cfg:
        parser.error("pass --load-packed or --cfg")

    config = None
    if args.cfg:
        argv = ["--cfg", args.cfg] + (["--mode", args.mode] if args.mode else [])
        config = build_config(argv, root_dir=ROOT)

    torch.manual_seed(0)
    source = args.load_packed or f"{args.cfg} (mode={config.mode})"

    # streaming and the side-by-side layout are exclusive: two columns cannot be filled
    # token by token at the same time, so a comparison prints once both answers are in
    streaming = not args.no_stream and not args.compare

    if args.compare:
        session = load_for_comparison(
            packed_path=args.load_packed, config=config, device=args.device,
            reference_device=args.reference_device, max_new_tokens=args.max_new_tokens,
        )
        session.follow_reference = args.follow_reference
        print(f"loaded {source}, against bf16 on "
              f"{args.reference_device or args.device}")
        print("greedy decoding both, so every difference is quantization error")
    else:
        model, tokenizer, cache_factory = load_for_chat(
            packed_path=args.load_packed, config=config, device=args.device
        )
        session = ChatSession(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            cache_factory=cache_factory,
            stream=streaming,
        )
        print(f"loaded {source}; KV cache {'quantized' if cache_factory else 'bf16'}")

    def answer(prompt):
        if args.compare:
            return session.ask(prompt).format(width=0)   # 0 -> the terminal's width
        if streaming:
            print("model> ", end="", flush=True)
            session.ask(prompt)      # the streamer already printed it
            return ""
        return f"model> {session.ask(prompt)}"

    if args.ask:
        text = answer(args.ask)
        if text:
            print(text)
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
        if prompt == "/summary" and args.compare:
            print(session.summary() or "(nothing asked yet)")
            continue
        print()
        text = answer(prompt)
        if text:
            print(text)


if __name__ == "__main__":
    main()
