#!/usr/bin/env python3
"""
MoE router 통계 수집기 — 프리필 시 layer별 '사용 expert 수'를 모델별/도메인별/입력 길이별로 측정합니다.

핵심 아이디어
  causal 모델은 앞쪽 L개 토큰의 라우팅이 뒤 토큰과 무관하므로, 최대 길이(예: 8192)로 한 번만 forward하고
  앞에서부터 128, 256, ... 토큰 구간의 고유 expert 수를 세면 모든 길이를 한 번에 얻을 수 있습니다.

지원 모델 (transformers >= 5.5 기준, router 모듈의 top-k 인덱스 출력을 hook)
  openai/gpt-oss-20b, Qwen/Qwen3.6-35B-A3B, google/gemma-4-26b-a4b (및 같은 구조의 MoE 모델)

사용 예
  # 1) 실제 측정 (GPU 서버)
  python moe_routing_stats.py --model Qwen/Qwen3.6-35B-A3B --out results --samples 32
  python moe_routing_stats.py --model google/gemma-4-26b-a4b --out results --samples 32
  python moe_routing_stats.py --model openai/gpt-oss-20b --out results --samples 32

  # 내 데이터 추가/대체 (jsonl의 "text" 또는 "messages" 필드, 또는 .txt 한 줄 = 한 문서)
  python moe_routing_stats.py --model ... --domains chat code --data service=/data/prompts.jsonl

  # 2) 이미 수집한 raw CSV로 분석만 다시
  python moe_routing_stats.py --analyze results/raw_Qwen3.6-35B-A3B.csv --out results

  # 3) 파이프라인 점검 (torch 불필요, 균등 라우팅 시뮬레이션 → 엑셀 공식과 비교)
  python moe_routing_stats.py --simulate --sim-E 256 --sim-k 8 --sim-layers 40 --out sim

출력 (--out 폴더)
  raw_<model>.csv                 domain, sample, length, layer별 사용 expert 수 (원자료)
  layer_stats_<model>.csv         domain × length × layer별 mean / p95 / max
  conservative_<model>.csv        length × layer 보수값 (도메인별 p95 중 최댓값)
  excel_paste_<model>.csv         length별 layer 평균 보수값  ← 엑셀 'Expert 실측' 시트에 붙여넣기
  eeff_<model>.csv                domain × layer별 유효 expert 수 E_eff = exp(entropy)
  heatmap_<model>.png             layer × length 사용 비율 히트맵 (보수값)
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]

# 기본 도메인: (datasets 이름, config, split, 필드). 접근이 안 되면 --data로 직접 지정하세요.
BUILTIN_DOMAINS = {
    "chat": ("HuggingFaceH4/ultrachat_200k", None, "test_sft", "messages"),
    "code": ("codeparrot/codeparrot-clean-valid", None, "train", "content"),
    "math": ("openai/gsm8k", "main", "test", ("question", "answer")),
    "wiki": ("Salesforce/wikitext", "wikitext-103-raw-v1", "test", "text"),
    "ko": ("wikimedia/wikipedia", "20231101.ko", "train", "text"),
}


# ----------------------------------------------------------------------------- 데이터
def iter_builtin(domain):
    from datasets import load_dataset

    name, cfg, split, field = BUILTIN_DOMAINS[domain]
    ds = load_dataset(name, cfg, split=split, streaming=True)
    for ex in ds:
        if isinstance(field, tuple):
            yield {"text": "\n".join(str(ex[f]) for f in field)}
        elif field == "messages":
            yield {"messages": ex["messages"]}
        else:
            t = ex[field]
            if t and t.strip():
                yield {"text": t}


def iter_file(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if "messages" in ex:
                    yield {"messages": ex["messages"]}
                else:
                    yield {"text": ex.get("text") or ex.get("prompt") or ""}
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield {"text": line.rstrip("\n")}


def encode_doc(tok, doc):
    if "messages" in doc and getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template(doc["messages"], tokenize=True, add_generation_prompt=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return list(ids)
    if "messages" in doc:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in doc["messages"])
    else:
        text = doc["text"]
    return tok(text, add_special_tokens=False)["input_ids"]


def build_samples(tok, docs, n_samples, length):
    """같은 도메인 문서를 이어 붙여 정확히 length 토큰짜리 샘플 n개를 만든다."""
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    sep = [tok.eos_token_id] if tok.eos_token_id is not None else []
    samples, buf = [], list(bos)
    for doc in docs:
        ids = encode_doc(tok, doc)
        if not ids:
            continue
        buf.extend(ids + sep)
        while len(buf) >= length:
            samples.append(buf[:length])
            buf = list(bos)  # 다음 샘플은 새 문서부터 (남은 꼬리는 버림)
            if len(samples) >= n_samples:
                return samples
    if len(samples) < n_samples:
        print(f"  [경고] 데이터가 부족해 {len(samples)}/{n_samples}개 샘플만 생성", file=sys.stderr)
    return samples


# ----------------------------------------------------------------------------- 모델 / hook
def moe_dims(config):
    tc = getattr(config, "text_config", None) or config
    E = next((getattr(tc, a) for a in ("num_local_experts", "num_experts", "n_routed_experts") if getattr(tc, a, None)), None)
    k = next((getattr(tc, a) for a in ("num_experts_per_tok", "experts_per_token", "top_k_experts", "moe_top_k") if getattr(tc, a, None)), None)
    if not E or not k:
        raise SystemExit("config에서 expert 수 / top-k를 찾지 못했습니다. MoE 모델인지 확인하세요.")
    return int(E), int(k)


def load_model(model_id, dtype, device_map):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=td, device_map=device_map)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText  # Qwen3.6 / Gemma 4 처럼 멀티모달 래퍼인 경우

        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=td, device_map=device_map)
    model.eval()
    return tok, model


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def find_routers(model, E):
    """(layer_idx, module, is_logits) 목록. 1순위: 클래스명에 Router 포함, 2순위: out_features==E인 gate/router Linear."""
    import torch.nn as nn

    found = {}
    for name, mod in model.named_modules():
        if "vision" in name or "audio" in name:
            continue
        m = LAYER_RE.search(name + ".")
        if not m:
            continue
        layer = int(m.group(1))
        cls = type(mod).__name__
        if "Router" in cls and layer not in found:
            found[layer] = (mod, False)
    if not found:
        for name, mod in model.named_modules():
            m = LAYER_RE.search(name + ".")
            if m and isinstance(mod, nn.Linear) and mod.out_features == E and re.search(r"(gate|router)$", name):
                found.setdefault(int(m.group(1)), (mod, True))
    if not found:
        raise SystemExit("router 모듈을 찾지 못했습니다. find_routers()에 모델 구조를 추가하세요.")
    return [(l, *found[l]) for l in sorted(found)]


def make_hook(store, layer, k, is_logits):
    import torch

    def hook(_mod, _inp, out):
        idx = None
        if not is_logits:
            outs = out if isinstance(out, (tuple, list)) else (out,)
            for t in reversed(outs):  # top-k 인덱스는 보통 마지막 출력 (int, 마지막 차원 = k)
                if torch.is_tensor(t) and not t.is_floating_point() and t.shape[-1] == k:
                    idx = t
                    break
            if idx is None:  # 인덱스를 안 돌려주는 router → logits에서 직접 top-k
                logits = next(t for t in outs if torch.is_tensor(t) and t.is_floating_point())
                idx = logits.topk(k, dim=-1).indices
        else:
            idx = out.topk(k, dim=-1).indices
        store[layer] = idx.reshape(-1, k).to("cpu", torch.int32).numpy()

    return hook


def collect(args):
    import torch

    tok, model = load_model(args.model, args.dtype, args.device_map)
    E, k = moe_dims(model.config)
    routers = find_routers(model, E)
    print(f"[모델] {args.model}: E={E}, k={k}, MoE layer {len(routers)}개 hook")
    store = {}
    handles = [mod.register_forward_hook(make_hook(store, l, k, il)) for l, mod, il in routers]
    lengths = sorted(args.lengths)
    Lmax = lengths[-1]
    device = next(model.parameters()).device

    domains = {d: (lambda d=d: iter_builtin(d)) for d in args.domains}
    for spec in args.data or []:
        dname, path = spec.split("=", 1)
        domains[dname] = (lambda p=path: iter_file(p))

    rows, freq = [], {}
    for dname, it in domains.items():
        if args.synthetic:
            g = np.random.default_rng(0)
            samples = [g.integers(0, tok.vocab_size, Lmax).tolist() for _ in range(args.samples)]
        else:
            print(f"[데이터] {dname}: {args.samples}개 × {Lmax} 토큰 구성 중")
            samples = build_samples(tok, it(), args.samples, Lmax)
        for si, ids in enumerate(samples):
            store.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=device), use_cache=False)
            for layer, idx in store.items():
                idx = idx[:Lmax]
                for L in lengths:
                    rows.append((dname, si, L, layer, int(np.unique(idx[:L]).size)))
                f = freq.setdefault((dname, layer), np.zeros(E, np.int64))
                f += np.bincount(idx.ravel(), minlength=E)[:E]
            print(f"  {dname} {si + 1}/{len(samples)}", end="\r")
        print()
    for h in handles:
        h.remove()
    return finish(args, short_name(args.model), E, k, rows, freq)


# ----------------------------------------------------------------------------- 시뮬레이션 (torch 불필요)
def simulate(args):
    """균등 라우팅을 흉내 낸 가짜 데이터로 분석 파이프라인과 엑셀 공식을 검증한다."""
    E, k, nl = args.sim_E, args.sim_k, args.sim_layers
    g = np.random.default_rng(0)
    lengths = sorted(args.lengths)
    rows, freq = [], {}
    for dname, skew in (("uniform", 0.0), ("skewed", 1.0)):
        w = g.dirichlet(np.ones(E) * (1.0 / skew)) if skew else np.ones(E) / E
        for si in range(args.samples):
            for layer in range(nl):
                # 토큰마다 서로 다른 k개를 가중치 w로 비복원 추출
                idx = np.stack([g.choice(E, k, replace=False, p=w) for _ in range(lengths[-1])])
                for L in lengths:
                    rows.append((dname, si, L, layer, int(np.unique(idx[:L]).size)))
                f = freq.setdefault((dname, layer), np.zeros(E, np.int64))
                f += np.bincount(idx.ravel(), minlength=E)
    return finish(args, f"sim_E{E}_k{k}", E, k, rows, freq)


# ----------------------------------------------------------------------------- 분석
def short_name(model_id):
    return model_id.rstrip("/").split("/")[-1]


def uniform_expected(E, k, L):
    return E * (1 - (1 - k / E) ** L)


def finish(args, name, E, k, rows, freq):
    os.makedirs(args.out, exist_ok=True)
    raw = pd.DataFrame(rows, columns=["domain", "sample", "length", "layer", "used"])
    raw.insert(0, "model", name)
    raw["E"], raw["k"] = E, k
    raw_path = os.path.join(args.out, f"raw_{name}.csv")
    raw.to_csv(raw_path, index=False)
    if freq:
        ee = []
        for (d, layer), f in sorted(freq.items()):
            p = f / max(1, f.sum())
            p = p[p > 0]
            ee.append((name, d, layer, float(np.exp(-(p * np.log(p)).sum())), E))
        pd.DataFrame(ee, columns=["model", "domain", "layer", "E_eff", "E"]).to_csv(
            os.path.join(args.out, f"eeff_{name}.csv"), index=False)
    analyze(raw, args.out)
    return raw_path


def analyze(raw, out):
    name = raw["model"].iloc[0]
    E, k = int(raw["E"].iloc[0]), int(raw["k"].iloc[0])
    g = raw.groupby(["domain", "length", "layer"])["used"]
    st = g.agg(mean="mean", p95=lambda s: float(np.percentile(s, 95)), max="max").reset_index()
    st["uniform"] = st["length"].map(lambda L: uniform_expected(E, k, L))
    st.insert(0, "model", name)
    st.to_csv(os.path.join(out, f"layer_stats_{name}.csv"), index=False)

    # 보수값: layer × length마다 도메인별 p95 중 최댓값
    cons = st.groupby(["length", "layer"])["p95"].max().unstack("layer")
    cons.to_csv(os.path.join(out, f"conservative_{name}.csv"))

    # 엑셀 붙여넣기용: layer 평균 (가중치 읽기량은 layer별 사용 수의 합에 비례하므로 평균이 정확)
    paste = pd.DataFrame({
        "length": cons.index,
        "measured_used_layer_avg": cons.mean(axis=1).round(2).values,
        "measured_used_layer_max": cons.max(axis=1).values,
        "uniform_expected": [round(uniform_expected(E, k, L), 2) for L in cons.index],
    })
    paste["measured_ratio"] = (paste["measured_used_layer_avg"] / E).round(4)
    paste["uniform_ratio"] = (paste["uniform_expected"] / E).round(4)
    paste.insert(0, "model", name)
    paste.to_csv(os.path.join(out, f"excel_paste_{name}.csv"), index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ratio = (cons / E).T  # layer × length
        fig, ax = plt.subplots(figsize=(1.1 * ratio.shape[1] + 3, 0.28 * ratio.shape[0] + 2))
        im = ax.imshow(ratio.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(ratio.shape[1]), [str(c) for c in ratio.columns])
        ax.set_yticks(range(ratio.shape[0]), [str(r) for r in ratio.index])
        ax.set_xlabel("input length (tokens)")
        ax.set_ylabel("layer")
        ax.set_title(f"{name}: used experts / E (p95, max over domains)\nuniform: "
                     + ", ".join(f"{L}={uniform_expected(E, k, L) / E:.1%}" for L in ratio.columns[:3]))
        for (y, x), v in np.ndenumerate(ratio.values):
            ax.text(x, y, f"{v:.0%}", ha="center", va="center", fontsize=6,
                    color="white" if v < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"heatmap_{name}.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("[참고] matplotlib이 없어 히트맵은 건너뜀")

    print(f"\n[결과] {name}  (E={E}, k={k})")
    print(paste.to_string(index=False))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF 모델 id 또는 로컬 경로")
    ap.add_argument("--out", default="routing_results")
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    ap.add_argument("--samples", type=int, default=32, help="도메인당 샘플 수")
    ap.add_argument("--domains", nargs="*", default=["chat", "code", "math", "wiki", "ko"],
                    help=f"기본 제공 도메인: {', '.join(BUILTIN_DOMAINS)}")
    ap.add_argument("--data", nargs="*", help="추가 도메인 이름=파일경로 (.jsonl 또는 .txt)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--synthetic", action="store_true", help="랜덤 토큰으로 hook 동작만 점검")
    ap.add_argument("--analyze", help="raw_*.csv로 분석만 다시 실행")
    ap.add_argument("--simulate", action="store_true", help="torch 없이 균등/편향 라우팅 시뮬레이션")
    ap.add_argument("--sim-E", type=int, default=256)
    ap.add_argument("--sim-k", type=int, default=8)
    ap.add_argument("--sim-layers", type=int, default=4)
    args = ap.parse_args()

    if args.analyze:
        os.makedirs(args.out, exist_ok=True)
        analyze(pd.read_csv(args.analyze), args.out)
    elif args.simulate:
        simulate(args)
    elif args.model:
        collect(args)
    else:
        ap.error("--model, --analyze, --simulate 중 하나가 필요합니다")


if __name__ == "__main__":
    main()
