"""Chat with the quantized model and the bf16 one at once, and see where they part.

An accuracy score says a combination lost 3% on some benchmark. It does not say what that
feels like to talk to, and this project has already found that those two questions have
different answers -- a combination reading +0.11% on perplexity changed 31% of its
generations. So: ask both models the same thing, print both replies, and report how far
they tracked each other.

Greedy decoding is deterministic, so every divergence is quantization error and nothing
else. Once two greedy paths split they rarely rejoin, which is why the position of the
first different token is reported next to the raw agreement: 90% agreement that begins at
token 2 is a different thing from 90% that begins at token 40.

Two ways to run it, and they answer different questions:

  independent (default)  each model keeps its own reply in its own history, so divergence
                         compounds exactly as it would for someone actually using it
  follow_reference       both histories get the bf16 reply, so every turn starts from the
                         same place and the comparison is of that turn alone
"""

from dataclasses import dataclass, field

from llmquant.eval.evaluate import generation_agreement
from llmquant.s5_chat.session import ChatSession

__all__ = ["ComparisonSession", "ComparisonTurn"]


@dataclass
class ComparisonTurn:
    """One question, both answers, and how far they agreed."""

    prompt: str
    reference: str
    candidate: str
    agreement: float
    exact: bool
    first_divergence: int | None  # token index, None if they never diverged
    reference_tokens: int
    candidate_tokens: int

    def format(self, width: int = 88) -> str:
        head = "identical" if self.exact else (
            f"{self.agreement:.0%} of tokens agree"
            + (f", first difference at token {self.first_divergence}"
               if self.first_divergence is not None else "")
        )
        lines = [f"  bf16      > {self.reference[:width]}",
                 f"  quantized > {self.candidate[:width]}",
                 f"  {head}"]
        return "\n".join(lines)


@dataclass
class ComparisonSession:
    """A ChatSession per model, asked the same things in lockstep."""

    candidate: ChatSession
    reference: ChatSession
    follow_reference: bool = False
    turns: list = field(default_factory=list)

    def reset(self):
        self.candidate.reset()
        self.reference.reset()
        self.turns.clear()
        return self

    def ask(self, prompt: str) -> ComparisonTurn:
        reference = self.reference.ask(prompt)
        candidate = self.candidate.ask(prompt)

        ref_ids = list(self.reference.last_reply_ids)
        cand_ids = list(self.candidate.last_reply_ids)
        stats = generation_agreement([ref_ids], [cand_ids])

        n = min(len(ref_ids), len(cand_ids))
        first = next((i for i in range(n) if ref_ids[i] != cand_ids[i]), None)
        if first is None and len(ref_ids) != len(cand_ids):
            first = n  # same prefix, one stopped earlier

        if self.follow_reference:
            # rewrite the candidate's last turn so both continue from the same place. Its
            # cache described the reply it actually produced, so it has to go.
            self.candidate.history[-1] = {"role": "assistant", "content": reference}
            self.candidate._cache = None

        turn = ComparisonTurn(
            prompt=prompt,
            reference=reference,
            candidate=candidate,
            agreement=stats["generation_agreement"],
            exact=stats["generation_exact_match"] == 1.0,
            first_divergence=first,
            reference_tokens=len(ref_ids),
            candidate_tokens=len(cand_ids),
        )
        self.turns.append(turn)
        return turn

    def summary(self) -> dict:
        """Agreement over the conversation so far."""
        if not self.turns:
            return {}
        return {
            "turns": len(self.turns),
            "mean_agreement": sum(t.agreement for t in self.turns) / len(self.turns),
            "identical_replies": sum(t.exact for t in self.turns),
        }


def load_for_comparison(packed_path=None, config=None, reference_model=None,
                        device="cuda", reference_device=None, **chat_kwargs):
    """Build both sessions: the quantized model, and a bf16 one to judge it against.

    `reference_device` exists because the two have to fit at once. Llama-3.2-1B is 2.4 GB
    plus 1.9 GB and fits easily; OLMoE-1B-7B is 12.9 GB in bf16 and does not fit beside
    anything, so its reference goes on the CPU and answers slowly rather than not at all.
    """
    from transformers import AutoTokenizer

    from llmquant.core.config import ModelArgs
    from llmquant.core.model import load_pretrained
    from llmquant.s5_chat.session import load_for_chat

    model, tokenizer, cache_factory = load_for_chat(
        packed_path=packed_path, config=config, device=device
    )
    if reference_model is None:
        if packed_path is not None:
            from llmquant.s3_pack import read_header

            reference_model = read_header(packed_path)["meta"]["model_id"]
        elif config is not None:
            reference_model = config.model
        else:
            raise ValueError("need a reference model id")

    reference, _ = load_pretrained(
        ModelArgs(model_id=reference_model, device=reference_device or device)
    )
    reference_tokenizer = AutoTokenizer.from_pretrained(reference_model)

    return ComparisonSession(
        candidate=ChatSession(model=model, tokenizer=tokenizer,
                              cache_factory=cache_factory, **chat_kwargs),
        reference=ChatSession(model=reference, tokenizer=reference_tokenizer, **chat_kwargs),
    )
