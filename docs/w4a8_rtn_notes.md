# W4A8 Symmetric RTN Quantization — 작업 정리

## ▶ 이어서 하기 (마지막 업데이트: 2026-09-19)

### 현재 위치
**Phase 0~4가 전부 통과했다. 남은 건 Phase 5(채팅)와 중단된 sweep 재개.**

| Phase | 상태 | 통과 근거 |
|---|---|---|
| 0 환경·baseline | ✅ | bf16 PPL 13.1642 |
| 1 fake quant | ⚠️ 조합 비교는 했으나 **마지막 sweep이 7/25에서 중단됨** | 아래 "바로 다음" 1번 |
| 2 real quant | ✅ | fake 13.2019 vs real 13.2027 (0.006%) |
| 3 packing | ✅ | 정수·scale 113/113 bit-exact, 로드 모델 로짓 완전 동일 |
| 4 커널 (weight-only) | ✅ | M≤4에서 Phase 2와 `torch.equal`, decode 1.11x, VRAM −34% |
| 5 채팅 | 미착수 | |

- **레이아웃이 단일 정의로 고정됐다** (`core/layout.py`). fake/real/pack/커널이 전부 여기서 읽는다.
  q/k/v는 out축 head별, o_proj는 in축 head별, 나머지는 끝 패딩. "레이아웃 명세" 절 참고.
- **정확도 기준은 생성 태스크**(ARC-Easy/Challenge/OpenBookQA), PPL은 보조. "선택 기준" 절.
- **가중치 우선순위: decode 4.0 > prefill 2.0 > accuracy 1.0**, BPV 0.5.
- 테스트 **243개** 통과. 커밋 6개, `origin/master`에 푸시 완료
  (https://github.com/pjh5672/llm-quant).

### 바로 다음에 할 일
1. **sweep 재실행** (25런, 약 1.5시간). 마지막 실행이 7/25에서 중단됐고, 그 뒤로 레이아웃과
   가중치가 바뀌어서 기존 결과는 무효다.
   ```powershell
   .\.venv\Scripts\python.exe examples\phase1_sweep.py --cfg configs\phase1\sweep.yaml
   ```
   중단분에서 본 것: **kv_cache int4가 정확도를 0.31 → 0.088로 무너뜨린다**(랜덤 추측보다 낮음).
   int4 weight와 겹치며 증폭된 것으로 보이고, 상호작용 분석이 정량화해줄 것이다.
2. Phase 5 채팅 (`stages/s5_chat/`) — packed 모델을 로드해서 대화. 로드 경로는 이미 동작한다.

### 결정할 것
1. **W4를 어떻게 할지.** ✅ 측정으로 결론: **보완책 없이는 int4를 어디에도 못 쓴다.**
   전부 int4 +29.31% / mlp만 +18.52% / attn만 +5.95%, 전부 5% 기준 밖.
   보완책을 만든다면 **MLP를 겨냥해야 한다** (손실의 대부분이 거기서 나온다).
2. **W8A8 vs W8A16.** ✅ **사실상 A16으로 결론.** Phase 4에서 **weight-only 커널만 만들었고**
   int-int(A8) 커널은 없다. 근거: `torch._int_mm`이 bf16 대비 1.03~1.04x뿐이고 decode는
   `M>16`을 요구해 아예 못 쓴다. **이득은 int 연산이 아니라 int weight를 직접 읽는 대역폭**이다.
   A8을 되살리려면 int-int 커널을 만들어야 하는데, 그 경우 **정수 합은 순서 무관이라
   레퍼런스와 bit-exact 검증이 가능하다**는 장점은 있다.
3. scale dtype fp32 유지 여부. g128이라 scale이 약 1.5MB → 약 30MB. 지금은 fp32 유지.
4. **head_weight.** 기준끼리 충돌한다 — 디스크는 bf16(int8이면 tie가 끊겨 +252MB),
   decode 속도는 int8(토큰마다 lm_head를 통째로 읽음, 1.62x → 1.94x). 실측 커널이 생겼으니
   이제 재볼 수 있다. 기본값은 bf16.

### 알아둘 것 (반복해서 부딪힌 것들)
- **1B 모델의 decode는 weight 대역폭 바운드가 아니다.** 토큰 시간의 45%만 weight 읽기라
  커널의 op 단위 2.37배가 전체에서는 1.11배가 된다. **모델이 클수록 이득이 커진다.**
- **`mode=fake`의 TTFT/TPS는 배포 수치가 아니다.** 모든 조합이 bf16 weight를 들고 있어
  구분이 안 된다. 판단은 `decode_gb_at_context`(해석적)로 한다.
- **PPL은 손상을 과소평가한다.** +0.11%로 보이는 조합이 생성의 31%를 바꾼다.
- **패딩은 수치적으로 공짜지만 용량은 아니다.** head_dim 64를 128 타일에 넣으면 attention
  4개 projection이 전부 2배가 되어 **int8 attention이 bf16과 같은 바이트**가 된다.

### 끝까지 확인하지 못한 것
- **TTFT가 fake 48ms → kernel 79ms로 나빠진다.** prefill마다 weight를 dequant하는 비용.
  없애려면 텐서코어 mainloop에 dequant를 fuse해야 한다(AWQ/Marlin 방식, 훨씬 큰 작업).
- `csrc/w4a8_rtn_naive.cu`, `build_and_run.bat`은 초기 group-wise 설계의 잔재. 현재
  `src/llmquant/stages/s4_kernel/`과 무관하므로 지워도 된다.

### 다시 시작하는 방법
```powershell
cd C:\Users\Park Jiho\Desktop\Project\DEV\llm-quant
.\.venv\Scripts\python.exe -m pytest tests -q                                       # 243개
.\.venv\Scripts\python.exe examples\auto_llm.py --cfg configs\phase1\w8a16.yaml     # 단일 실행
.\.venv\Scripts\python.exe examples\phase1_sweep.py --cfg configs\phase1\sweep.yaml # 전체 ~1.5시간
.\.venv\Scripts\python.exe examples\phase4_bench_gemm.py                            # GEMM 속도
.\.venv\Scripts\python.exe examples\phase1_analyze.py experiments\phase1-sweep\sweep.json --bpv-weight 5
```
- `mode`: `fake`(정확도 측정) | `real`(오라클) | `kernel`(배포). `pack: true`는 `mode=kernel` 필요.
- 패키지 재설치: `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` (ninja 포함)
- PowerShell에서 `$env:PYTHONIOENCODING="utf-8"` 권장.
- CUDA 커널은 첫 호출 때 자동 JIT 빌드(1~2분), 이후 캐시.
- 결과는 `experiments/<project>/`에 저장됨 (config 복사 + `result.json` / `sweep.json` / `model.bin`).

---

## 확정 설계

**목표**: 성능 저하를 파악한 최적 bit 조합으로 CUDA 커널 기반 가속 추론 + 간단한 채팅까지.

| 대상 | 방식 | 선택지 |
|---|---|---|
| weight (lm_head 제외 Linear) | static, 오프라인 symmetric RTN, **group_size=128** (K축) | int4 / int8 |
| lm_head weight | static, 오프라인 symmetric RTN, group_size=128 | int8 / bf16 |
| activation (lm_head 입력 포함) | dynamic, symmetric, **per-token × group_size=128** (K축) | int8 / bf16 |

- 캘리브레이션 없음. activation은 저장할 scale이 없고 config(bits, dynamic, per-token, group_size, symmetric)만 저장.
- 출력 수식(A8일 때): `y[m,n] = Σ_g s_x[m,g] · s_w[n,g] · Σ_{k∈g} a[m,k]·q[n,k]`
  → **group(128)마다 int32 누적을 끊고 float로 부분합을 합산.**
- per-channel은 `group_size = K`(group 1개)인 특수 경우로 계속 표현 가능.
- **activation per-channel(dynamic)은 기각**: int GEMM으로 계산할 수 없어 가속 이점이 없음. prefill에서 미래 토큰 정보가 scale에 섞여 causal 구조가 깨짐. batch나 decode 조건에 따라 결과가 달라져 성능 저하 보장을 검증할 수 없음.
- per-token int8 손실이 크면 Hadamard rotation(QuaRot 방식, 데이터 불필요)을 옵션 조합으로 추가해 비교.

### group 128 설계 (2026-09-18 확정)

**축**: weight `[N, K]`는 K(in_features)축을 128씩. activation `[..., K]`도 **같은 K축**을 128씩.
즉 activation scale은 `[토큰, K/128]`로, **각 토큰의 scale이 그 토큰 값만 보고 정해짐.**

- 기각했던 "activation per-channel"(토큰 축으로 묶기)의 문제 — prefill에서 미래 토큰이 scale에
  섞여 causal 구조가 깨지는 것, batch에 따라 결과가 달라지는 것 — 은 **여기 해당되지 않음.**
  causal·batch-invariant 유지됨.
- weight group과 축·경계가 같아야 int GEMM이 성립함. 그래서 축 선택지는 사실상 K축뿐.
- **패딩 정책 (2026-09-19)**: 마지막 축이 128의 배수가 아니면 **뒤에 0을 붙여** 채운다.
  zero padding은 symmetric abs-max에서 **scale을 바꾸지 못하므로**(0은 `max(|x|)`에 영향 없음)
  실제 원소는 짧은 그룹으로 양자화한 것과 **비트 단위로 같다.** 바뀌는 건 회계(그룹 수)뿐이다.
  덕분에 "어디서나 group 128"이라는 규칙 하나로 통일된다.
- **o_proj만 head 단위로 그룹을 끊는다.** o_proj의 입력은 attention 출력을 이어붙인 것이라
  K축이 `heads × head_dim`(32 × 64)이다. 평범한 128 그룹은 **두 head를 한 그룹에 섞어** 서로
  다른 dynamic range가 scale을 공유하게 만든다. head별로 쪼개고 64를 128로 패딩한다.
  - 효과(실측, o_proj 입력): 평균 절대오차 **0.1026 → 0.0922 (약 10% 감소)**
  - 비용: o_proj scale이 2배(그룹 16 → 32). 전체 디스크 0.9708 → 0.9728 GB (+2MB)
  - **weight와 activation 양쪽 모두** head 단위로 끊어야 int GEMM의 부분합 분해가 성립한다.
- 그 외 Linear의 in_features는 2048 또는 8192라 128로 나누어떨어져 패딩이 발생하지 않는다.

**영향**
- 크기: decoder Linear 973M개 → scale 376K개에서 7.6M개로. fp32 기준 약 1.5MB → 약 30MB (W4 모델 대비 +3%).
- Phase 2가 **쉬워짐**: group당 부분합 최대치가 `128 × 127 × 127 ≈ 2.06M`으로 fp32의 정확 정수 한계
  (2^24 ≈ 16.7M) 안. 즉 **W8A8도 `torch._int_mm` 없이 fp32 matmul로 bit-exact**하게 계산됨 (TF32는 꺼야 함).
  기존에 적어둔 "W8A8은 int32나 fp64 필요" 제약은 group 128에서는 사라짐.
- Phase 4 커널은 **어려워짐**: K 방향 128마다 누적을 끊어야 함. QQQ가 per-channel GEMM과
  per-group GEMM을 별도 구현한 이유가 이것. 재사용 후보를 볼 때 이 점을 반영할 것.


## 단계 순서

보장 대상은 fake quant가 아니라 **real quant 결과**. fake와 real의 차이는 float 정밀도 수준인지 보는 sanity check.

| Phase | 내용 | 통과 조건 |
|---|---|---|
| 0 | 환경, bf16 baseline | ✅ PPL 13.1642 |
| 1 | **fake quant**(bf16 dequant) 조합 비교 → bf16 대비 성능 저하 → 최적 조합 선택 | ✅ 완료 (9조합 + attn/mlp 16조합) |

| 2 | **real quant** reference: packing 안 한 int weight + group별 정수 누적 | ✅ **통과** — fake 13.2019 vs real 13.2027 (차이 0.006%) |
| 3 | **packing → bin 파일 저장 → Python 로드** | ✅ **통과** — 정수·scale 113/113 bit-exact, 로드 모델 로짓 완전 동일 |
| 4 | **custom CUDA 커널**이 int weight를 직접 읽어서 연산 | ✅ weight-only 완료 — M≤4에서 Phase 2와 `torch.equal`. decode 1.11x, VRAM −34% |
| 5 | 채팅 | 정상 대화 |

- **최적 조합 선택 기준**: bf16 대비 PPL 증가 5% 이내(≈13.82 이하) 조합 중 모델 크기가 가장 작은 것. (A8/A16 구분이 안 되는 문제 있음 → 위 "결정할 것" 2번)
- **Phase 2 대상**: 선택한 조합 + PPL 차이가 1% 이내인 차순위 조합.
- **bit-exact 규칙**: 아래 "bit-exact 규칙 (확정)" 절 참고. quant-dequant 부분은 이미 커널로 검증 완료.
- **fake quant의 정확한 계산 한계**: group 128에서는 group당 부분합이 최대 약 206만이라
  W4A8·W8A8 **모두** fp32로 정확하게 계산됨(2^24 이내, TF32는 꺼야 함).
  per-channel(group = K)이었을 때의 "W8A8은 int32나 fp64 필요" 제약은 group 128에서는 해당 없음.

## bit-exact 규칙 (확정, 2026-09-18)

레퍼런스는 `llmquant.utils.quant_ops.fake_quantize`. 커널은 이걸 **그대로** 재현해야 함
(`torch.equal`로 검증, `tests/test_cuda_kernel.py`).

1. **scale은 fp32.** `amax`는 fp32로 올린 뒤 계산.
2. **`amax / qmax`는 참(true) IEEE 나눗셈.**
   ⚠️ 함정: PyTorch에서 `tensor / 7`처럼 **파이썬 스칼라로 나누면 역수 곱셈**으로 내려가서
   참나눗셈과 **1 ulp** 어긋난다. 커널에서는 재현 불가능.
   그래서 레퍼런스는 반드시 **텐서로 나눈다**: `amax / torch.tensor(float(qmax))`.
   (이 한 줄 때문에 초기 커널 검증에서 그룹의 약 49%가 불일치했음.)
3. **`clamp(min=SCALE_EPS)`** where `SCALE_EPS = 1e-8` → CUDA `fmaxf(v, 1e-8f)`.
4. **반올림은 half-to-even.** `torch.round` ↔ CUDA **`rintf`**.
   ⚠️ `roundf`는 half-away-from-zero라 **쓰면 안 됨**.
5. **clamp 순서**: `clamp(q, -qmax-1, qmax)` ↔ `fminf(fmaxf(q, qmin), qmax)`.
6. **dequant 후 입력 dtype으로 캐스팅**은 round-to-nearest-even (`c10::BFloat16`, `__float2bfloat16`).
7. 빌드에 **`--use_fast_math`를 쓰지 말 것.** 나눗셈이 근사 역수로 바뀌어 2번이 깨진다.

## Windows 빌드 환경 (해결, 2026-09-18)

> 이전 노트의 "`csrc/*.cu`가 한 번도 컴파일에 성공하지 못함 / nvcc 에러 메시지를 받아보지 못함"의 원인.

**근본 원인: `%TMP%`에 공백이 있으면 nvcc가 아무 메시지도 없이 exit 2로 죽는다.**
이 PC의 TMP는 `C:\Users\Park Jiho\AppData\Local\Temp`라 항상 걸렸다. 에러가 안 찍히니
원인 파악이 안 됐던 것. `TMP`/`TEMP`를 8.3 단축 경로(`C:\Users\PARKJI~1\...`)로 바꾸면 해결됨.

`src/llmquant/kernels/build.py`가 빌드 전에 자동으로 처리하는 것들:
1. **MSVC 환경 주입** — `cl.exe`가 PATH에 없어서 `vcvarsall.bat`을 실행해 env를 가져옴.
   ⚠️ `subprocess`에 **리스트가 아니라 문자열**로 넘겨야 함. 리스트로 넘기면 `list2cmdline`이
   따옴표를 `\"`로 escape해서 cmd가 배치 파일을 못 찾는다. (`_short_path`도 같은 함정)
2. **`TMP`/`TEMP` 8.3 변환** — 위 근본 원인.
3. **`TORCH_EXTENSIONS_DIR` 8.3 변환** — ninja도 공백에 약함.
4. **VS Installer 디렉터리를 PATH에 추가** — `vcvarsall.bat`이 내부에서 맨 이름 `vswhere.exe`를 호출함.
5. **`--use-local-env`** — torch 2.11이 Windows에서 이 플래그를 안 붙여줘서, nvcc가 스스로
   `vcvars64.bat`을 다시 돌리려다 "Could not set up the environment"로 실패한다.

**소스 구조 주의**: `.cu`에서 `torch/extension.h`를 include하면 안 됨.
`compiled_autograd.h`가 CUDA의 `cuda::std`와 충돌해 MSVC에서 `error C2872: 'std' 모호한 기호`가 난다.
→ 커널은 `.cu`(ATen 헤더만), pybind 바인딩은 `.cpp`로 분리.

**확인된 조합**: nvcc 12.8 + MSVC 14.44.35207 + torch 2.11.0+cu128 + sm_120, ninja 필요(`pip install ninja`).

## 환경 / 모델

- **서버**: 이 Windows 10 PC. RTX 5060 Ti 16GB (sm_120), `llm-quant/.venv` (Python 3.13.2, torch 2.11.0+cu128, transformers 5.17.0), nvcc 12.8 + MSVC 14.44 (VS2022 BuildTools, `cl.exe`는 PATH에 없어서 `vcvarsall.bat` 필요).
- QQQ와 vLLM은 Linux 기준이라 이 PC에서는 재사용이 어려울 수 있음.
- **모델**: `meta-llama/Llama-3.2-1B-Instruct` (로컬 HF 캐시에 있음, HF 토큰 설정됨). hidden 2048, intermediate 8192, 16 layers, vocab 128256, bias 없음, **tie_word_embeddings=true**.
- **평가**: wikitext-2-raw-v1 test, seqlen 2048, 비중첩 141개 구간 (로컬 캐시에 있음).

## Config 설계 (gaia-compressor 매핑, 2026-09-18 확정)

gaia-compressor에서 **구조와 개념만** 가져오고 구현은 이 프로젝트에 맞게 다시 씀.

### 가져온 것
- YAML(`defaults/quantization/evaluation/sweep`) + CLI override. 단, **CLI가 YAML을 이김**
  (gaia는 반대로 YAML이 CLI를 덮어씀 — 이건 의도적으로 바꿈).
- 타깃별 data type 설정, `experiments/<project>/`에 config 복사 + 결과 저장.
- `auto_llm.py` 선형 파이프라인 형태.

### 안 가져온 것과 이유
- **gaia의 packing 포맷** — GAIA NPU 전용(`GAIA_DIM_SIZE` 재배열, 1024 chunk 안 bit-serialize,
  GMX super/sub exponent). CUDA 커널이 못 읽음. 아래 "packing 저장 형식"을 그대로 유지.
- **모델별 어댑터 18종** — attention 재작성까지 포함해 무거움. `QuantizationModifier`로 충분.
- **pruning / AWQ·GPTQ·SmoothQuant·SpinQuant** — 캘리브레이션 없는 RTN이 전제. 슬롯만 남김.
- **config DSL의 `-g[128]-rw-zp` 접미사** — group_size는 전역 한 개, symmetric·K축 고정이라
  표현할 게 없음. `int4` / `int8` / `bf16` 세 값만 씀.
- `QMatmul`(attention BMM 양자화), `act_out`, easydict.

### gaia에 없어서 새로 만들어야 하는 것
**로드 경로.** gaia는 packing이 NPU로 내보내는 **단방향**이고 packed를 다시 읽는 코드가 없음.
"packed 모델 로드 → 런타임"은 베낄 게 없는 신규 작업 (Phase 4, `load_packed`).

### config 형태

```yaml
defaults:
  project: llama3.2-1b-w4a8
  model: meta-llama/Llama-3.2-1B-Instruct
  device: cuda
  seq_len: 2048
  seed: 0
  save_path: null       # 미구현 (Phase 3)
  pack: false           # 미구현 (Phase 3)
  load_packed: null     # 미구현 (Phase 4)

quantization:
  quantize: true
  quant_method: rtn     # rtn만 구현
  mode: fake            # fake | real(Phase 2) | kernel(Phase 4)
  attn_weight: int4     # self_attn.{q,k,v,o}_proj
  mlp_weight: int4      # mlp.{gate,up,down}_proj
  head_weight: bf16     # lm_head
  activation: int8      # 위 Linear들의 입력
  kv_cache: bf16        # 미구현 (아래 참고)
  group_size: 128       # weight와 activation이 공유

evaluation:
  metric: wikitext2
  generation: true        # stage 2 실행 여부
  generation_task: arc_easy   # arc_easy | gsm8k | null
  generation_task_limit: null # null이면 태스크 기본값 (arc 500, gsm8k 100)
  lambada_limit: 500
  max_new_tokens: 64

sweep:                  # 선택. 있으면 교차곱으로 확장
  attn_weight: [int4, int8]
  mlp_weight: [int4, int8]
```

### 규칙
- **`bf16` = 그 타깃을 양자화하지 않음.** YAML `null`과 키 생략도 같은 뜻.
- **weight는 `int4`/`int8`/`bf16`, activation은 `int8`/`bf16`만.** activation int4는 확정 설계 밖이다:
  출력 수식과 packing 포맷이 int8 activation을 전제로 잡혀 있고, int4×int4 GEMM은 CUTLASS로도
  QQQ로도 공짜로 안 나오는 별도 커널 경로다. config 단계에서 `ValueError`로 막는다.
- **weight가 bf16이면 그 타깃의 activation도 bf16.** bf16 weight에 int8 activation을 먹이면
  돌릴 int GEMM이 없어서 이득이 없음. (기존 lm_head 동작을 일반화한 것)
- **activation은 weight와 같은 group_size를 강제로 공유.** 경계가 어긋나면 int GEMM의
  부분합 분해가 성립하지 않음. 그래서 group_size는 타깃별이 아니라 전역 한 개.
- **embed는 축에 없음.** `tie_word_embeddings=true`지만 `FakeQuantLinear.from_linear`가
  새 파라미터를 만들어 **tie를 끊기 때문에** embedding은 bf16으로 남음. `size.py`도 그 전제로 계산함.
- **미구현 슬롯은 조용히 무시하지 않고 `NotImplementedError`를 냄** (`pack`, `load_packed`,
  `mode=real/kernel`, `quant_method != rtn`). "설정했는데 아무 일도 안 일어나는"
  상태를 만들지 않기 위함.

### kv_cache (2026-09-19 구현됨)
전에는 "PPL로는 효과가 0이라" 미뤘었다. **정확도 지표를 생성 태스크로 바꾸면서 측정이 가능해졌다** —
`evaluate_ppl`은 청크마다 forward 한 번이라 캐시를 되읽지 않지만, 생성 태스크는 디코드 루프를 돈다.

**구현**: `stages/s1_fake/fake_quant_cache.py::FakeQuantCache` — `DynamicCache`를 상속해
K/V를 쓰는 시점에 fake quant한다. `generate(past_key_values=...)`로 주입.

- **그룹은 128, 64짜리 head는 패딩해서 채운다.** 위 패딩 정책과 같은 규칙이고, 결과는
  head_dim(64)로 그룹을 끊은 것과 **비트 단위로 동일**하다. KV 양자화의 표준 granularity
  (토큰별·헤드별)와도 일치한다.
- **이미 캐시에 있는 토큰은 다시 양자화하지 않는다.** 재양자화하면 스텝마다 오차가 누적된다.

**비용은 디스크가 아니라 컨텍스트에 비례한다** (Llama-3.2-1B, `decode_bytes_at_context`):

| kv dtype | KV KB/token | ctx 2k | ctx 8k | ctx 32k |
|---|---|---|---|---|
| bf16 | 32.0 | 1.486 GB | 1.674 GB | 2.424 GB |
| int8 | 17.0 | 1.457 GB | 1.557 GB | 1.955 GB |
| int4 | 9.0 | 1.442 GB | 1.494 GB | **1.705 GB** |

디스크 크기는 **전혀 안 변한다**(캐시는 저장 대상이 아님). 2k에서는 weight가 지배해서 이득이 3%뿐이고,
32k에서 30%가 된다. 그래서 점수의 decode 항은 `context_tokens`(기본 2048) 기준으로 계산한다.

**정확도 실측** (ARC-Easy n=200, attn·mlp int8): bf16 0.565 / int8 0.585 / **int4 0.430**.
int8은 사실상 무료, **int4는 −13.5%p로 못 쓴다.**

## 프로젝트 구조

**core / stages / eval 3층.** phase별 폴더를 쓰되, phase에 진짜로 속하는 것만 나눴다 —
지금 코드의 대부분은 여러 phase가 공유하므로 억지로 가르면 오히려 찾기 어려워진다.

- **`core/`** — 모든 stage가 공유. config, scheme, scale 계산, quant-dequant 연산,
  모델을 갈아끼우는 modifier, 비용 지표. **stage를 module 레벨에서 import하지 않는다.**
- **`stages/`** — phase당 폴더 1개. `mode` config 값과 1:1로 대응해서
  "mode=real이면 어떤 코드가 도는가"를 grep 없이 알 수 있다. 아직 없는 phase의 폴더는
  거기 무엇이 들어올지와 통과 조건을 docstring으로 적어뒀다.
- **`eval/`** — 측정과 판단. PPL, LAMBADA, 생성, GEMM 속도, 분석.

```
llm-quant/
├── pyproject.toml, .gitignore
├── docs/w4a8_rtn_notes.md                  # 이 파일
├── configs/phase1/                         # bf16-baseline, w4a8, w8a16, sweep
├── src/llmquant/
│   ├── __init__.py                         # QuantizationModifier, QuantConfig, oneshot, evaluate_ppl
│   ├── core/
│   │   ├── config.py                       # ModelArgs, DatasetArgs, QuantConfig(attn/mlp/head/activation/kv_cache/group_size)
│   │   ├── parser.py                       # RunConfig, build_config(YAML+CLI), expand_sweep
│   │   ├── selection.py                    # SelectionConfig (PPL 제약 + 가중치)
│   │   ├── scheme.py                       # QuantizationArgs, QuantizationScheme, PRESET_SCHEMES
│   │   ├── modifier.py                     # QuantizationModifier(scheme, attn_scheme, mlp_scheme, lm_head_scheme, mode)
│   │   ├── layout.py                       # WeightLayout — 패딩·그룹·scale 기하의 단일 정의
│   │   ├── observers.py                    # compute_scale (group / channel / token, fp32)
│   │   ├── quant_ops.py                    # quantize, dequantize, fake_quantize ← 레퍼런스
│   │   ├── metrics.py                      # model_metrics: 디스크 / decode 트래픽 / BPV
│   │   ├── model.py, oneshot.py
│   │   └── datasets/                       # wikitext, lambada, prompts, tasks(arc_easy/gsm8k)
│   ├── stages/
│   │   ├── __init__.py                     # quant_linear_for(mode) — core가 지연 import
│   │   ├── s1_fake/fake_quant_linear.py    # Phase 1  mode="fake"
│   │   ├── s2_real/real_quant_linear.py    # Phase 2  mode="real" — 정수 weight, group별 정수 누적
│   │   ├── s3_pack/                        # Phase 3  packing/.bin  (예정)
│   │   ├── s4_kernel/                      # Phase 4  mode="kernel"
│   │   │   ├── build.py                    #   MSVC env + TMP/8.3 + ninja, load_extension()
│   │   │   ├── ops.py                      #   fake_quantize_cuda(x, args, return_scale=False)
│   │   │   ├── quant_linear.py             #   KernelQuantLinear — M으로 커널/cuBLAS 디스패치
│   │   │   └── csrc/fake_quant.cu, wq_gemv.cu, fake_quant.cpp
│   │   └── s5_chat/                        # Phase 5  채팅          (예정)
│   └── eval/
│       ├── evaluate.py                     # ppl / lambada / 생성태스크 / 일치도 / measure_latency
│       ├── run.py                          # run_one, run_generation — 단일 실행과 sweep의 공통 경로
│       ├── analysis.py                     # pareto / axis_effects / interactions / 가중 점수
│       ├── report.py                       # 종합 표 렌더링 (계산과 분리)
│       └── benchmark.py                    # GEMM 벤치 (bf16 vs int8 vs int8 g128)
├── examples/
│   ├── auto_llm.py                         # config 1개 실행 (phase 무관)
│   ├── phase1_sweep.py                     # 그리드 확장 + 2단계 평가 + 분석 리포트
│   ├── phase1_analyze.py                   # 저장된 sweep.json 재분석 (GPU 불필요)
│   ├── phase1_generation.py                # 끝난 sweep에 stage 2(생성 평가)만 얹기
│   └── phase4_bench_gemm.py                # GEMM 속도 측정
├── tests/                                  # 243개
│   ├── test_quantization.py  11            ├── test_cuda_kernel.py  41 (bit-exact)
│   ├── test_config.py        21            ├── test_parser.py       27
│   ├── test_analysis.py      15            ├── test_metrics.py       8 (디스크 vs decode 충돌)
│   ├── test_tasks.py         24 (채점 파싱) ├── test_report.py        8
│   ├── test_grouping.py      13 (패딩/head) ├── test_kv_cache.py      8
│   ├── test_real_quant.py    16 (Phase 2)   ├── test_kernel_linear.py 18 (Phase 4)
│   ├── test_packing.py       16 (Phase 3)   └── test_benchmark.py      4
├── csrc/w4a8_rtn_naive.cu, build_and_run.bat   # 초기 naive 커널 잔재 (지워도 됨)
├── results/                                # git 제외 (구 결과 보관)
└── experiments/<project>/                  # git 제외, config 복사 + result.json / sweep.json
```

**의존 방향은 `core → stages`가 한 군데뿐이다**: `modifier.apply()`가 `mode`에 맞는 Linear
클래스를 `stages.quant_linear_for()`로 받아온다. 이걸 module 레벨에서 import하면
core가 stage에 묶여버리므로 함수 안에서 지연 import한다.

사용 예시:
```bash
python examples/auto_llm.py --cfg configs/phase1/w4a8.yaml
python examples/auto_llm.py --cfg configs/phase1/w4a8.yaml --mlp-weight int8   # CLI가 이김
python examples/phase1_sweep.py --cfg configs/phase1/sweep.yaml
```
```python
from llmquant import QuantConfig, oneshot
recipe = QuantConfig(attn_weight="int8", mlp_weight="int4", activation="int8").to_modifier()
oneshot(model, recipe)
```
`QuantizationModifier(scheme="W8A8")` 프리셋 경로도 그대로 동작함(모든 Linear에 같은 scheme).

- `mode`는 현재 `"fake"`만 있음. Phase 2에서 `"real"`(`modules/real_quant_linear.py`), Phase 4에서 `"kernel"`(`modules/kernel_quant_linear.py`)을 추가.
- 해당 Phase에서 추가할 것: `compression/`(packing, bin 형식), `kernels/`, `entrypoints/chat.py`, `examples/quantize_and_save.py`.
- **lm_head 동작**: `lm_head_scheme=None`이면 weight와 activation 모두 bf16 유지. bf16 weight에 int8 activation을 쓰는 커널은 이점이 없어서 초기 스크립트 동작에서 바꿈.
- recipe(YAML), pipelines, calibration 흐름은 필요 없어서 가져오지 않음.

## Phase 1 결과 (group_size=128, 2026-09-18)

`results/scheme_sweep.json`. 크기는 packing weight + fp32 scale(그룹당 1개), 나머지는 bf16으로 계산한 이론값.
per-channel 기준 구 결과는 `results/scheme_sweep_per_channel.json`, `results/phase1_fake_quant_legacy.json`에 보관.

| 조합 | PPL | bf16 대비 | 크기(GB) | 5% 이내 | (참고) per-channel |
|---|---|---|---|---|---|
| bf16 | 13.1642 | 0 | 2.302 | - | - |
| W4A8, lm_head int8 | 17.0370 | +29.42% | 1.223 | ✗ | +177.5% |
| W4A8, lm_head bf16 | 17.0323 | +29.38% | 0.971 | ✗ | +177.4% |
| W4A16, lm_head int8 | 17.0248 | +29.33% | 1.223 | ✗ | +186.6% |
| W4A16, lm_head bf16 | 17.0223 | +29.31% | 0.971 | ✗ | +186.6% |
| W8A8, lm_head int8 | 13.1930 | +0.22% | 1.676 | ✓ | +1.31% |
| W8A8, lm_head bf16 | 13.1914 | +0.21% | 1.424 | ✓ | +1.31% |
| W8A16, lm_head int8 | 13.1811 | +0.13% | 1.676 | ✓ | +0.30% |
| **W8A16, lm_head bf16** | 13.1783 | +0.11% | 1.424 | ✓ (기준상 선택) | +0.32% |

### attn / mlp 분리 결과 (16조합, `experiments/phase1-sweep/sweep.json`)

구 9조합과 겹치는 **9개가 전부 bit 단위로 동일하게 재현됨** (config 파이프라인 검증 완료).

| attn | mlp | head | act | PPL | bf16 대비 | 크기(GB) | 5% 이내 |
|---|---|---|---|---|---|---|---|
| int4 | int4 | bf16 | bf16 | 17.0223 | +29.31% | 0.971 | ✗ |
| int4 | int8 | bf16 | bf16 | 13.9481 | **+5.95%** | 1.346 | ✗ (아깝게) |
| int8 | int4 | bf16 | bf16 | 15.6020 | **+18.52%** | 1.049 | ✗ |
| **int8** | **int8** | **bf16** | **bf16** | **13.1783** | **+0.11%** | **1.424** | ✓ (선택) |
| int8 | int8 | bf16 | int8 | 13.1914 | +0.21% | 1.424 | ✓ |

(head=int8 행은 모두 크기 +252MB에 PPL도 미세하게 나빠서 생략. 전체는 sweep.json 참고)

**손실이 어디서 오는지 분리됨 — 직관과 반대다**
- **mlp만 int4로 내리면 +18.52%**, **attn만 int4로 내리면 +5.95%**.
  → **MLP가 attention보다 int4에 훨씬 민감하다.** (attention이 더 민감할 것이라는 예상은 틀렸음)
- MLP는 decoder Linear 파라미터의 **82.8%**를 차지하면서 **동시에 더 민감한** 부분이다.
  즉 "mlp만 int4로 내려 크기를 벌자"는 전략은 **가장 나쁜 교환**이다
  (크기 1.424 → 1.049GB, 26% 절약에 PPL +18.4%p).
- 반대로 attn만 int4로 내리는 건 손실은 작지만(+5.95%p) 크기 이득도 작다(1.424 → 1.346GB, 5.5%).
  그마저 5% 기준(13.82)을 넘긴다.
- 두 손실은 대략 가산적이다 (5.95 + 18.52 ≈ 24.5 vs 실측 29.31, 약간 초가산).
- **결론: 보완책 없이는 int4를 어디에도 쓸 수 없다.** 보완책을 만든다면 **MLP를 겨냥**해야 한다.

**activation 양자화 비용은 weight 조합과 무관하게 일정하다**

| weight 조합 | act bf16 | act int8 | 비용 |
|---|---|---|---|
| int4/int4 | 17.0223 | 17.0323 | +0.07%p |
| int4/int8 | 13.9481 | 13.9647 | +0.13%p |
| int8/int4 | 15.6020 | 15.6237 | +0.16%p |
| int8/int8 | 13.1783 | 13.1914 | +0.10%p |

A8은 어떤 weight 조합 위에서도 **+0.07~0.16%p**만 낸다. int8 GEMM 가속이 목표라면 이 정도는
사실상 공짜이고, 이것이 결정 2번에서 W8A8을 미는 근거다.

**축끼리 상호작용하는가** (`analysis.py`의 interactions, 기준 = attn-int8/mlp-int8/head-bf16/act-bf16)

| 같이 바꾼 축 | 예측(단일 효과 합) | 실측 | residual |
|---|---|---|---|
| attn + mlp | 3.1935 | 3.8439 | **+0.6504 (+16.9%)** |
| mlp + activation | 2.4367 | 2.4454 | +0.0086 (+0.4%) |

- **weight끼리는 강하게 상호작용한다.** attn과 mlp를 같이 int4로 내리면 따로 내렸을 때의 합보다
  17% 더 아프다. 즉 **weight 조합은 축별로 따로 판단할 수 없다.**
- **activation은 사실상 독립이다** (residual +0.4%). weight를 뭘로 하든 A8 비용이 일정하다는
  위 표와 일치한다. → **A8 결정은 weight 결정과 분리해서 내려도 된다.**
- `attn_weight: int8 -> int4`의 비용 폭이 +5.85~+10.79%p로 넓은 것도 같은 현상이다
  (mlp가 int8이냐 int4냐에 따라 달라짐). 반면 activation의 폭은 +0.08~+0.18%p로 좁다.

**결과에서 보이는 점**
- **g128은 양쪽 다 크게 개선했다.** W4 +177% → +29%, W8A8 +1.31% → **+0.22%**.
- **그래도 W4 per-group RTN은 여전히 못 쓴다 (+29.3%).** 5% 기준(13.82)에 한참 못 미침.
  group 세분화만으로는 해결이 안 되고, W4를 쓰려면 **보완책이 필수**다
  (weight만 보는 채널별 clipping ratio 탐색(MSE 최소화), 또는 Hadamard rotation).
  Llama-3.2-1B이 작은 모델이라 양자화 내성이 낮은 것으로 보인다.
- **W8A8과 W8A16의 차이가 0.10%p로 줄었다** (per-channel일 때는 1.01%p).
  A8을 써도 정확도 손해가 사실상 없다는 뜻 → int8 GEMM 가속이 목표라면 A8 선택 근거가 강해짐.
- **A8과 A16의 크기가 여전히 같다.** activation scale은 저장하지 않으므로 당연한 결과이고,
  "5% 이내 중 가장 작은 모델" 기준으로는 A8이 영원히 선택될 수 없다. (아래 "결정할 것" 2번)
- lm_head는 bf16이 크기(1.424 < 1.676)도 PPL도 더 낫다. int8로 할 이유가 없음
  (tie_word_embeddings 때문에 크기 이득이 없고 오히려 +252MB).
- **아직 안 본 영역: attn과 mlp를 다르게 주는 조합.** 이 모델은 decoder Linear 파라미터의
  **82.8%가 MLP**(레이어당 50.33M / 60.82M)다. `mlp_weight: int4` + `attn_weight: int8` 같은
  조합이 크기 이득 대부분을 가져가면서 정확도를 지킬 수 있는지는 측정된 바 없다.
  구 sweep은 이걸 표현할 수 없었고, `configs/phase1/sweep.yaml`(16런)이 바로 이 영역을 덮는다.

## sweep 분석 (`llmquant/analysis.py`)

sweep 결과를 받아 **bit 조합 결정에 필요한 비교**를 자동으로 만들어 낸다. 전부 순수 함수라
저장된 `sweep.json`만 있으면 GPU도 모델도 없이 다시 돌릴 수 있다:

```bash
python examples/phase1_analyze.py experiments/phase1-sweep/sweep.json --limit-ratio 1.05
```

리포트 구성:
1. **전체 런 표** — PPL 순 정렬, 선택된 조합에 `*` 표시.
2. **selection** — 기준 안에서 가장 작은 모델 + **near miss**(아깝게 놓친 조합).
   기준을 조정할 가치가 있는지 판단하려면 near miss가 꼭 필요하다.
3. **pareto front** — "더 작으면서 동시에 더 정확한" 조합이 없는 것들. 기준과 무관하게
   실제로 고려할 가치가 있는 후보만 남는다.
4. **axis effects** — **한 축만 바꾸고 나머지를 고정**했을 때의 비용. 여러 context에서 측정한
   평균과 **min..max 폭**을 같이 낸다. **폭이 넓다는 것 자체가 그 축이 다른 축과 상호작용한다는 신호다.**
5. **interactions** — 여러 축을 동시에 바꾼 조합에서 **실측 손실 vs 단일 축 손실의 합**.
   residual > 0이면 같이 바꿀 때 더 아프다는 뜻이고, 그러면 축별로 따로 판단할 수 없다.

**기준점(reference)에 주의**: 4·5번의 기준은 bf16 baseline이 **아니라 그리드 안에서 가장 정밀한 조합**이다.
baseline은 그리드 밖에 있어서, 모든 런이 attn·mlp를 둘 다 양자화하면 "한 축만 다른 행"이
존재하지 않아 분해가 시작되지 않는다. (실제로 처음 구현했을 때 interactions가 0개로 나왔다.)

## 속도 측정 (2026-09-18) — ⚠️ A8 전제가 흔들림

### 왜 fake quant 경로를 재지 않는가
`FakeQuantLinear`는 weight를 로드 시점에 한 번 dequant해서 **bf16으로 들고 있고**, forward는
평범한 bf16 `F.linear`다. 그래서 fake 경로에서는:
- **W8A16**: forward에 추가 연산 없음
- **W8A8**: forward마다 activation을 fake quant → **A16보다 느림**

즉 fake quant 모델의 속도를 재면 배포 시와 **정반대** 결론이 나온다. 대신 **GEMM 연산 자체**를
재면 fake quant 오버헤드에 오염되지 않은 숫자가 나오고, Phase 4를 기다릴 필요도 없다.

```bash
python examples/phase4_bench_gemm.py --m 1 64 512 2048
```

### 측정 결과 (RTX 5060 Ti, sm_120, torch 2.11+cu128, TF32 off)

| shape | M | bf16 | TFLOPS | int8 full-K | int8 g128(torch 조합) |
|---|---|---|---|---|---|
| gate/up_proj | 2048 | 1.452ms | 47.3 | **1.04x** | 0.05x |
| gate/up_proj | 8192 | 5.543ms | 49.6 | **1.03x** | 0.05x |
| down_proj | 2048 | 1.488ms | 46.2 | 0.99x | 0.09x |
| q/o_proj | 2048 | 0.397ms | 43.3 | 1.03x | 0.10x |
| 모든 shape | 1 | - | - | **측정 불가** | - |

**읽는 법**
- **`torch._int_mm`은 bf16 대비 1.03~1.04x밖에 안 된다.** M=8192까지 키워도 같다.
  bf16이 48~50 TFLOPS로 포화된 compute 바운드 구간이므로 공정한 비교다.
- **decode(M=1)는 int8 GEMM을 아예 쓸 수 없다**: `torch._int_mm`이 `M > 16`을 요구한다.
  게다가 decode는 메모리 바운드라 속도가 weight 바이트 수로 결정되고, A8/A16은 크기가 같다.
  → **decode에서 A8의 이득은 원리적으로 0이다.**
- **g128을 torch 연산으로 조합하면 0.05x로 파국적이다.** group마다 커널 런치가 들어가기 때문
  (K=8192면 64회). 설계의 실제 속도가 아니라 **하한**이고, Phase 4 커널이 반드시 fuse해야 한다는
  근거로만 읽을 것.

### 결정 2번에 미치는 영향
W8A8을 선택할 근거는 **오직 int8 GEMM 속도**였는데, **이 GPU에서 지금 쓸 수 있는 경로로는
그 이득이 3~4%뿐이다.** 반면 비용은 PPL +0.10%p로 확정돼 있다.

⚠️ 단, 이건 `torch._int_mm`(cuBLAS 래퍼)을 잰 것이지 튜닝된 커널이 아니다. sm_120에서
torch의 int8 경로가 텐서코어를 제대로 못 쓰고 있을 가능성이 있고, CUTLASS/Marlin 계열
커널이라면 달라질 수 있다. 그걸 확인하는 게 Phase 4다.
**다만 입증 책임이 A8 쪽으로 넘어갔다.** 기본값은 W8A16으로 두는 편이 안전하다.

## 선택 기준 (2026-09-19 개정) — 정확도는 생성 태스크, 순위는 가중 점수

### 왜 PPL을 주 지표에서 내렸나
PPL은 teacher-forced 단일 forward라 **생성 경로를 한 번도 안 탄다.** 실측으로
`int8/int8`이 PPL +0.11%인데 **생성의 31%가 bf16과 다르고** LAMBADA는 −2%p 떨어졌다.
PPL은 여전히 싸고 논문 비교가 되므로 **보조 지표로 남기되**, 기준이 최적화하는 값은 아니다.

정확도 = **생성 태스크 스위트의 평균 정확도**, 비용 = **bf16 대비 상대 저하율(%)**.
태스크 점수가 없는 옛 sweep 행은 PPL 증가율로 자동 대체되므로 과거 결과도 그대로 분석된다.

### 표에 같이 놓는 네 축
| 축 | 지표 | 출처 |
|---|---|---|
| 정확도 | 태스크 평균 + 저하율 `dacc%` | 생성 (ARC-Easy/Challenge, OpenBookQA, GSM8K) |
| 정확도(보조) | **PPL** + `dPPL%` | teacher-forced |
| 압축 | **BPV** = `num_bits + scale_bits/group_size` | 해석적 |
| decode 속도 | `decode_gb_per_token` + **proj** 배속 | 해석적 (decode는 메모리 바운드) |
| (선택) 지연 | TTFT ms, decode TPS | 실측 — **기본 off**, 아래 참고 |

BPV 예: int4 g128 = **4.25**, int8 g128 = **8.25**, bf16 = **16**.

⚠️ **실측 TTFT/TPS는 기본에서 껐다 (`latency: false`).** fake quant는 weight를 dequant해서
bf16으로 들고 있어 **모든 조합의 decode 트래픽이 같다** — 측정 TPS가 조합 간에 거의 안 움직이고
A8만 activation 양자화 오버헤드로 느려 보인다. 즉 **표에 두면 오해만 부른다.**
속도 신호는 **BPV와 `decode_gb_per_token`/`proj`**가 담당한다.
Phase 3~4에서 `latency: true`로 켜면 그때 실측이 의미를 갖는다 (하네스는 그대로 재사용).
켜져 있을 때만 열이 나타나고, 리포트가 이 경고를 표 아래에 직접 찍는다.

### lm_head에서 기준이 충돌한다 (실측)
| 조합 | 디스크 | decode GB/token | proj | BPV |
|---|---|---|---|---|
| bf16 | 2.3019 | 2.3019 | 1.00x | 16.00 |
| attn·mlp int8, head **bf16** | **1.4240** | 1.4240 | 1.62x | 9.90 |
| attn·mlp int8, head **int8** | 1.6762 | **1.1870** | **1.94x** | 8.25 |
| attn·mlp int4, head int8 | 1.2231 | 0.7338 | 3.14x | 5.10 |

`tie_word_embeddings=true`라 lm_head를 양자화하면 tie가 끊겨 **디스크는 +252MB 늘지만**,
decode는 lm_head를 토큰마다 통째로 읽고 embedding은 행 조회뿐이라 **트래픽은 줄어든다.**
→ **head_weight를 크기만 보고 확정하면 안 된다.**

### 가중 점수
```yaml
selection:
  acc_drop_limit_pct: 5.0   # 하드 제약: 정확도 저하율 상한 (null이면 해제)
  accuracy_weight: 1.0
  bpv_weight: 2.0
  decode_speed_weight: 2.0
  prefill_speed_weight: 0.0   # latency를 실제로 재는 단계에서만 의미가 있음
```
점수 = `w_bpv·BPV이득% + w_decode·트래픽이득% + w_prefill·TTFT이득% − w_acc·정확도저하%`
(전부 bf16 대비 %라 가중치가 서로 비교 가능하다).

- 하드 제약을 통과한 것 중 점수 최대를 고른다.
- 가중치를 바꿔가며 재분석: `phase1_analyze.py <sweep.json> --bpv-weight 5 --no-limit`
- ⚠️ **없는 지표를 0으로 치지 않는다.** 정확도는 baseline에서 직접 유도하고, 나머지는
  `score_missing`에 기록한다. (0으로 치면 가장 공격적인 조합이 항상 이긴다 — 테스트가 잡은 버그)

이전 기준은 "PPL 5% 이내 중 **가장 작은 모델**"이었다. 두 가지 문제가 있었다.
1. **속도를 전혀 못 본다.** A8과 A16은 크기가 같아서 기준상 A8이 영원히 선택될 수 없다.
2. **디스크 크기는 속도의 프록시로 틀리다.** lm_head에서 둘이 정면으로 어긋난다 (아래).

## 생성 평가 (2026-09-18)

PPL은 teacher-forced라 **생성 경로를 한 번도 안 건드린다.** 디코드 루프와 KV 캐시를 쓰는
지표가 따로 필요하다.

### 지표들
- **생성 태스크 정확도** (`generation_task`) — **실제로 토큰을 생성해서 정답과 맞추는 절대 지표.**
  PPL·LAMBADA는 teacher-forced 단일 forward고, 생성 일치도는 상대 지표라 "얼마나 나빠졌나"를
  못 말한다. 이 지표만 그 둘을 동시에 만족한다.

  | 태스크 | 출처 | n(기본) | 생성 길이 | config당 | 노이즈 |
  |---|---|---|---|---|---|
  | **`arc_easy`** (기본) | `allenai/ai2_arc` ARC-Easy | 500 | 8토큰(정답 letter) | ~30초 | ±4%p |
  | `gsm8k` | `openai/gsm8k` | 100 | 256토큰(CoT) | ~3분 | ±9%p |

  GSM8K가 긴 CoT라 오차 누적을 더 잘 잡지만, 1B 모델은 점수가 낮아(~30%) 감당 가능한 n에서
  **노이즈가 신호보다 크다.** 그래서 기본은 ARC-Easy.

  ⚠️ **채점 파싱이 조용히 틀리기 쉬운 지점이다.** ARC는 영어 관사 `a`가 선택지 `A`로 오인될 수
  있어서 **대소문자 일치를 먼저 시도**하고 없을 때만 fallback한다. GSM8K는 `####` 뒤 숫자를
  우선 보고, 없으면 마지막 숫자로 떨어진다. 둘 다 `tests/test_tasks.py`로 고정.

- **LAMBADA 마지막단어 정확도** — 로컬 캐시(`EleutherAI/lambada_openai`, 5153개).
  절대 수치. teacher-forced지만 **토큰 하나만 틀려도 예제 전체가 오답**이라 PPL보다 민감하다.
- **bf16 대비 생성 일치도** — 고정 프롬프트 16개를 greedy 생성해서 bf16 출력과 비교.
  `generation_agreement`(토큰 일치율), `generation_exact_match`, `generation_first_divergence`.
  라벨이 필요 없다. greedy는 결정적이라 어긋남은 전부 양자화 오차다.
  **"달라졌다"를 재지 "나빠졌다"를 재지 않으므로**, 가중 점수에서는 태스크 정확도가 있으면
  그쪽을 쓰고 없을 때만 이걸 쓴다.

### 2단계 sweep
전체 그리드는 PPL(런당 ~1분)로 거르고, **Pareto front + 기준 통과 조합에만** 생성 평가를
돌린다. 17조합 전부에 돌리면 1시간을 넘긴다.

stage 1은 생성 지표를 추가해도 안 변하므로, **이미 끝난 sweep에 stage 2만 얹을 수 있다**:
```bash
python examples/phase1_generation.py experiments/phase1-sweep/sweep.json
python examples/phase1_generation.py experiments/phase1-sweep/sweep.json --generation-task gsm8k
```

### 스모크 결과 — ⚠️ PPL이 손상을 과소평가한다
(LAMBADA 100개, 32토큰 — 본 실행보다 작은 설정. n=100은 노이즈 ±5%p)

| 조합 | PPL | LAMBADA | bf16 일치율 | 완전 일치 |
|---|---|---|---|---|
| bf16 | 13.1642 | 0.530 | - | - |
| int8/int8 | +0.11% | 0.510 | **0.925** | **0.875** |
| int4/int8 | +5.95% | 0.480 | 0.508 | 0.250 |
| int4/int4 | +29.31% | 0.390 | 0.257 | 0.063 |

- `int8/int8`은 PPL +0.11%로 "사실상 무손실"로 보이지만 **프롬프트의 12.5%에서 bf16과 다른
  텍스트를 생성**하고, LAMBADA는 **−2.0%p(상대 −3.8%)** 떨어진다. PPL 증가폭의 30배가 넘는다.
- **해석 주의**: 일치율은 "나빠졌다"가 아니라 "달라졌다"를 잰다. greedy라 토큰 하나가 틀어지면
  이후가 전부 어긋나 민감하다. **절대 품질은 LAMBADA를 봐야 한다.**
- 본 실행에서는 `lambada_limit: 500`으로 돌릴 것.

## 지연 측정 (`llmquant/eval/evaluate.py::measure_latency`)

prefill과 decode를 **따로** 잰다. 서로 다른 regime이라 한쪽만 좋아지는 선택이 있기 때문이다.

- **TTFT** = 프롬프트 제출부터 첫 토큰까지. `max_new_tokens=1` 생성 시간.
- **decode TPS** = `(전체 시간 − TTFT) / (N−1)`. `min_new_tokens`를 걸어 매 반복 길이를 고정하고,
  중앙값을 취한다.
- **peak VRAM**도 같이 기록한다 (fake 모드에서는 조합 간 차이가 없지만 Phase 3~4에서 의미가 생긴다).

⚠️ **GPU가 한가할 때 재야 한다.** 다른 작업과 같은 GPU를 쓰면 값이 무의미하게 낮아진다.

## 레이아웃 명세 (2026-09-19 확정) — `core/layout.py`

**fake-quant → real-quant → pack → 추론이 전부 같은 정의를 써야 정합성이 보장된다.**
그래서 기하(패딩·그룹·scale 개수)를 `llmquant/core/layout.py::WeightLayout` 한 곳에 모았고,
각 단계는 거기서 읽는다. 흩어져 있으면 packing과 커널이 어긋나도 **조용히 틀린 값**이 나온다.

### 규칙

| module | reduction(in) 축 | output 축 |
|---|---|---|
| q/k/v_proj | 128 배수로 **끝 패딩** | **head별 분할**, head_dim을 128 배수로 패딩 |
| o_proj | **head별 분할**, head_dim을 128 배수로 패딩 | 128 배수로 끝 패딩 |
| mlp, lm_head | 128 배수로 끝 패딩 | 128 배수로 끝 패딩 |

q/k/v는 head를 **만들어내서** out축에 head 구조가 있고, o_proj는 head를 **소비해서** in축에 있다.
head별로 끊으면 dynamic range가 다른 두 head가 scale을 공유하지 않는다.

**scale 위치**: q/k/v는 **output head당 1개**(`out_group = head_dim`), 나머지는 output 행당 1개.
reduction 축으로는 전부 그룹당 1개.

### 비용 — ⚠️ head_dim이 64라 attention이 전부 2배가 된다

패딩 자체는 **수치적으로 공짜**다(0은 symmetric abs-max를 못 바꾸므로 실제 원소는 짧은 그룹으로
양자화한 것과 비트 단위로 같다). 공짜가 아닌 건 **저장 용량**이다.

| module | [out, in] | 패딩 후 | 슬롯 | int8 vs bf16 |
|---|---|---|---|---|
| q_proj | [2048, 2048] | [**4096**, 2048] | 2.00x | **1.00x** |
| k/v_proj | [512, 2048] | [**1024**, 2048] | 2.00x | **1.00x** |
| o_proj | [2048, 2048] | [2048, **4096**] | 2.00x | **1.00x** |
| mlp | [8192, 2048] | [8192, 2048] | 1.00x | 0.50x |

**int8 attention은 bf16과 바이트가 똑같다.** 64짜리 head를 128 타일에 넣으면 행(또는 열)이
2배가 되는데 int8의 비트 절감도 정확히 2배라 서로 상쇄된다. int4는 4배가 아니라 2배만 절감한다.
MLP는 head 구조가 없어서 영향이 없다.

| 조합 | 디스크 | BPV | (q/k/v out패딩 껐을 때) |
|---|---|---|---|
| bf16 | 2.3019 GB | 16.00 | - |
| attn·mlp int8 | 1.5793 GB | 10.977 | 1.4884 GB |
| attn·mlp int4 | 1.0480 GB | 7.284 | 1.0040 GB |

**BPV는 패딩을 센다** (`bits_per_real_element`). 실제 원소당 비트로 계산하지 않으면 디스크가
90MB 움직여도 BPV가 그대로라 지표가 거짓말을 한다.

### 정확도 비용 (out축 head 단위 scale)
q/k/v의 scale을 행당 1개에서 **head당 1개**로 묶으면 scale이 64배 줄고(32768 → 512)
weight 오차가 커진다. 실측:

| module | int8 | int4 |
|---|---|---|
| q_proj | 2.71x | 2.62x |
| k_proj | 2.37x | 2.32x |
| v_proj | 1.56x | 1.55x |

`qkv_out_scale_per_head: false`로 끄면 레이아웃 정렬은 유지하면서 이 정확도 비용만 없앨 수 있다.

## Phase 2 real quant (2026-09-19 완료) — `stages/s2_real/`

**qdq가 아니다.** s1_fake는 dequant한 bf16 weight를 들고 평범한 bf16 matmul을 돌려서 정확도 비용만
잰다. 여기서는 weight를 **정수로 들고 정수로 곱한다** — Phase 4 커널이 재현해야 할 산술이다.

    A16  group마다 weight를 dequant해서 fp32로 누적
    A8   activation을 토큰·group별로 양자화 → group 안에서 int32 누적 → scale 곱 → 부분합 합산

    y[m,n] = Σ_g  s_x[m,g] · s_w[n,g] · Σ_{k∈g} a[m,k]·q[n,k]

### 통과 조건 ✅
| | fake | real | 차이 |
|---|---|---|---|
| wikitext2 PPL (W8A8) | 13.2019 | 13.2027 | **0.006%** |

레이어 단위로 보면 **real이 fake보다 일관되게 정확하다**(W8A16 기준 오차 0.00212 → 0.00195).
fake는 양쪽을 bf16으로 반올림하고 거기서 누적하는데, real은 정수를 정확히 누적하고 group당 한 번만
scale을 곱하기 때문이다. **둘이 같아야 하는 게 아니라 real이 조금 더 정확한 게 맞다.**

### 왜 fp32 matmul로 충분한가
group 128이면 부분합 최대치가 `128 × 127 × 127 = 2,064,512`로 fp32의 정확 정수 범위(2^24) 안이다.
따라서 **정수 값에 대한 fp32 matmul은 곧 정수 연산**이고 int32 경로가 필요 없다.
K=8192 전체를 누적하면 1.3억까지 올라가 그 범위를 벗어나는데, 이것도 그룹을 축 끝까지 끌지 않고
128에서 끊는 이유 중 하나다.

### ⚠️ TF32 관련 기존 서술 정정
노트에 "TF32는 꺼야 함"이라고 적어뒀었는데 **측정해보니 int8 입력에서는 영향이 없다.**
TF32의 가수는 11비트이고 `|int8| ≤ 127 < 2^11`이라 입력이 정확히 표현되며, 누적은 어차피 fp32다.
실제로 최악 케이스(±127로 채운 group)에서도 TF32 on/off 결과가 동일했다.
TF32가 깨뜨리는 건 **11비트를 넘는 입력**일 때뿐이다(4097을 넣으면 128만큼 틀림).
`exact_fp32_matmul()` 가드는 그래서 **보험이지 필수가 아니다** — 입력이 11비트를 넘게 되면 그때
필수가 된다.

### 속도
real은 fake보다 **3.5배 느리다**(PPL 76s → 268s). group마다 matmul을 따로 돌리기 때문이고,
**이게 바로 Phase 4 커널이 하나로 fuse해야 하는 이유다.** 레퍼런스의 목적은 속도가 아니라
커널을 검증할 오라클을 만드는 것이다.

## Phase 4 커널 (2026-09-19) — `stages/s4_kernel/`

### 무엇을 만들었나
`csrc/wq_gemv.cu` — **weight-only 양자화 matmul.** int8 weight를 메모리에서 그대로 읽어
레지스터에서 dequant하고, activation은 bf16으로 둔다. 커널은 `[N, groups, 128]` **canonical
형태**만 받으므로 head 구조를 전혀 모른다 — 패딩과 head 분할은 `core/layout.py`가 이미 끝냈다.

### 왜 weight-only인가 (int-int가 아니라)
측정 결과 **`torch._int_mm`은 bf16 대비 1.03~1.04배**뿐이고 decode(M=1)는 int8 GEMM을 아예 못 쓴다.
즉 이득은 int 연산이 아니라 **weight를 int로 직접 읽는 대역폭**에서 나온다.

### M으로 디스패치 (crossover는 측정값)
| M | kernel | fake | |
|---|---|---|---|
| **1 (decode)** | 0.013 ms | 0.030 ms | **2.37x 빠름** |
| 2 | 0.017 | 0.018 | 1.06x |
| 4+ | 0.025~ | 0.018 | 느림 |

M=1에서 4MB/0.013ms = **308 GB/s**(피크의 69%). M이 커지면 compute 바운드라 텐서코어를 쓰는
cuBLAS가 이기므로, `KERNEL_MAX_ROWS=4` 위에서는 weight를 한 번 dequant해서 cuBLAS에 넘긴다.
dequant는 O(N·K)라 M행에 걸쳐 상각된다.

### 정확도 — 두 경로가 각각 무엇과 일치하는가
- **M ≤ 4 (커널 경로): Phase 2 레퍼런스와 `torch.equal`로 완전히 일치.** 허용치가 아니라 등식이다.
- **M > 4 (cuBLAS 경로): fake quant와 오차가 정확히 같다.** 둘 다 bf16으로 dequant 후 bf16
  matmul이라 같은 계산이다. 즉 대체한 경로보다 나쁘지도 좋지도 않다.

네 가지 레이아웃(plain / int4 / o_proj head_dim / q_proj out_group) 전부에서 확인했다.

### 전체 모델 실측 (Llama-3.2-1B, W8 attn·mlp·head)
| | TTFT | decode | peak VRAM |
|---|---|---|---|
| fake | 48.3 ms | 74.7 tok/s | 2.91 GB |
| **kernel** | 79.1 ms | **82.7 tok/s (1.11x)** | **1.91 GB (−34%)** |
| bf16 | 58.0 ms | 77.9 tok/s | 2.42 GB |

### ⚠️ op 단위 2.37배가 전체에서 1.11배가 된 이유
**1B 모델의 decode는 weight 대역폭 바운드가 아니다.** 측정 대역폭 384 GB/s 기준으로:

| | weight/token | 이론 최소 | 실측 토큰시간 | weight 비중 |
|---|---|---|---|---|
| bf16 | 2.302 GB | 6.00 ms | 13.25 ms | **45%** |
| kernel int8 | 1.342 GB | 3.50 ms | 13.32 ms | 26% |

토큰 시간의 45%만 weight 읽기이고 나머지는 커널 런치·attention·프레임워크 오버헤드다.
weight를 절반으로 줄여도 **상한이 1.23배**이고 실측 1.11배는 그 안이다.
**모델이 클수록 weight 비중이 커지므로 이 커널의 이득도 커진다.**

### 남은 것
- **TTFT가 48 → 79ms로 나빠진다.** prefill마다 weight를 dequant하는 비용이다. 이걸 없애려면
  텐서코어 mainloop에서 dequant를 fuse해야 하는데(AWQ/Marlin 방식) 훨씬 큰 작업이다.
  decode 우선순위가 1번, prefill이 2번이라 현재 트레이드오프는 우선순위와 맞다.
- **int-int(A8) 커널**은 아직 없다. weight-only와 달리 **정수 합은 순서 무관**이라
  레퍼런스와 bit-exact 검증이 가능하다는 장점이 있다. 커널이 group 합을 먼저 내고 scale을
  나중에 곱하는 구조를 유지한 것이 이 때문이다.
- 첫 시도 기록: M 루프를 weight 로드 바깥에 두어 weight를 M번 다시 읽었고 M=64에서 0.01배였다.
  group마다 블록 전체 리덕션을 돈 것도 병목이었다. warp당 group 1개 + int32로 4개씩 읽기 +
  마지막에 리덕션 1회로 바꿔 0.052 → 0.013ms가 됐다.

## Phase 3 packing (2026-09-19 완료) — `stages/s3_pack/`

### 통과 조건 ✅
| | 결과 |
|---|---|
| qweight bit-exact | **113/113** |
| wscale bit-exact | **113/113** |
| 로드한 모델의 로짓 | **원본과 완전히 동일** (max\|diff\| 0.00) |

W4 전체 양자화 기준 파일 **1.1312 GB**, 저장 1.2초 / 로드 2.3초. 로드 시 **bf16 weight는
한 번도 만들어지지 않는다** — 구조는 model_id의 config에서 오고, 양자화 레이어는 packed 정수에서
바로 만들어지며, 양자화하지 않은 텐서만 bf16으로 읽는다.

### canonical 레이아웃을 저장한다
논리적 `[out, in]`이 아니라 **커널이 그대로 읽는 `[N, groups, 128]`** (패딩·head 분할 완료)을
저장한다. 논리 모양을 저장하고 로드 때 reshape하면 패딩과 head 경계에 합의해야 하는 곳이
**두 군데**가 되고, 언젠가 어긋나면 증상이 에러가 아니라 **틀린 숫자**다.

### 디버깅에서 나온 함정 3개
전부 "weight는 bit-exact인데 로짓이 다르다"로 나타났다. packing 자체는 처음부터 맞았다.

1. **non-persistent 버퍼.** RoPE의 `inv_freq`는 `state_dict()`에 없는데, 로더는 meta에서
   모델을 만들고 `to_empty()`를 부른다 — 그러면 그 버퍼가 **초기화되지 않은 메모리**를 가리킨다.
   빼먹으면 position encoding이 쓰레기를 읽는다 (max|diff| 14.4).
   → 저장하되 **원래 dtype으로**. `inv_freq`를 bf16으로 저장하면 위치를 표현할 정밀도가 안 된다.
2. **파생 버퍼.** 양자화 레이어는 scale에서 파생된 버퍼(`_broadcast_scale`)도 들고 있는데,
   이걸 일반 루프에서 bf16으로 저장하니 **생성자가 정확히 계산해둔 값을 로더가 덮어썼다**
   (max|diff| 0.81). → 양자화 레이어 소유 텐서는 **prefix로 통째 제외**. 나중에 파생 버퍼를
   하나 더 추가해도 같은 버그가 재발하지 않는다.
3. `state_dict()`를 쓰면 1번이 자동으로 빠지는데, 그게 오히려 함정이었다.

### 형식
```
[magic "LLMQUANT"][header 길이 8byte][JSON header][64byte 정렬된 raw tensor bytes]
```
- **nibble packing**: int4 두 개를 1byte. `+8`로 0~15 편향 후 짝수 index는 하위 4bit.
  `-3, 5` → `5, 13` → `0xD5` (노트 예시와 일치, 테스트로 고정). int8은 그대로.
  `+8` 편향은 **부호 확장 처리를 양쪽이 합의할 필요를 없앤다** — 몇 달 뒤 이 파일을 보고
  커널을 짜는 사람이 값 범위를 반으로 날려먹기 딱 좋은 지점이다.
- **64byte 정렬**: memory-map해서 정렬 복사 없이 GPU로 넘기기 위함.
- JSON header: tensor별 dtype·shape·offset·크기, 레이어별 양자화 args(head_dim·out_group 포함),
  non-persistent 버퍼 목록, 원본 model_id.

### config 연결
```yaml
defaults:
  pack: true              # mode=kernel 필요 — packed 레이아웃은 커널이 읽는 그 형태다
  load_packed: model.bin  # 있으면 양자화를 건너뛰고 로드
```
`pack: true`는 `experiments/<project>/model.bin`에 저장한다.
`pack`에 `mode != kernel`을 주면 **아무것도 못 읽는 파일이 나오므로 config 단계에서 거부**한다.

## packing 저장 형식 (확정)

단일 `.bin` = `[magic+version][header 길이 8byte][JSON header][정렬된 raw tensor bytes]`.
- **nibble packing**: int4 두 개를 1byte에 넣음. 각 값에 +8을 해서 0~15로 만든 뒤, 짝수 index는 하위 4bit, 홀수 index는 상위 4bit. 예: `-3, 5` → `5, 13` → `0xD5`. int8 weight는 packing 없이 그대로 저장.
- JSON header: tensor별 이름·dtype·shape·offset·크기, weight 정보(bits, per-channel, symmetric, scale dtype, packing 규칙), activation 정보(bits, dynamic, per-token, symmetric — tensor 없음), 원본 모델 ID.
- embedding, RMSNorm 등 양자화하지 않는 weight도 bf16으로 넣음 → bin 하나로 모델 전체 로드. tokenizer와 config는 모델 ID로 HF에서 가져옴.
- Python 로더: `np.memmap` → torch → GPU에 packing된 uint8 그대로 올림. `unpack_int4()`는 검증과 디버깅에만 씀. 추론할 때는 unpack 없이 커널만 packing 데이터를 읽음.

## CUDA 커널

### 완료: quant-dequant 커널 (2026-09-18)

`src/llmquant/kernels/csrc/fake_quant.cu` — fused symmetric RTN quant-dequant, 마지막 축 기준 group.

- `fake_quantize_cuda(x, args, return_scale=False)` → PyTorch 레퍼런스와 **bit-exact** (`torch.equal`).
- group 하나당 block 1개(128 thread), warp shuffle로 abs-max 리덕션 → scale → quant → dequant를 한 번에.
- bf16 / fp16 / fp32, 임의의 group_size(마지막 축을 나누어떨어지게), `strategy="channel"`(group 1개)도 지원.
- 속도: PyTorch 경로 대비 **7~9배** (2048×2048 bf16 기준 0.44ms → 0.062ms, 약 220~270 GB/s).
- 검증: `tests/test_cuda_kernel.py` 41개 — dtype·shape·group_size 조합, 전부 0인 그룹(eps 경로),
  극소/극대값, `.5` tie, non-contiguous 입력 포함.
- 최적화 여지: 현재 입력을 두 번 읽음(리덕션 1회 + 양자화 1회). group_size ≤ blockDim이면
  레지스터에 캐싱해 한 번만 읽을 수 있음. 지금은 병목이 아니라 보류.

### 남음: Phase 4 GEMM 커널

CUTLASS 등을 이용해 CUDA 커널을 빌드함. Phase 4 시작할 때 확인할 것:
- CUTLASS의 sm_120 지원 여부
- Windows(MSVC)에서 빌드되는지 (공식 지원은 주로 Linux)
- int8 GEMM 말고 packing된 int4를 직접 읽는 경로가 있는지 (없으면 그 부분은 직접 작성)
- **group 128이라 K 방향 128마다 누적을 끊어야 함** — QQQ 기준으로 per-group GEMM은 별도 구현 대상
- Python 연결: `torch.utils.cpp_extension` (`kernels/build.py`의 `load_extension()`을 그대로 확장하면 됨)

## lm_head (확정)

tie_word_embeddings=true라서 int8로 해도 크기 이득이 없음(+약 245MB). 정확도가 기준 안에 들면 Phase 4에서 bf16과 int8 속도를 비교해 결정. Phase 1의 크기 기준 선택은 decoder Linear와 activation 조합에만 적용.

---

## (초기 설계 — 참고용) GPU 없던 환경의 group-wise 설계

> 아래는 확정 설계 이전 내용. group-wise, 캘리브레이션 없는 W4A8 커널을 전제로 했으며 per-channel 설계로 대체됨.

### 참고 레포
- **mit-han-lab/llm-awq** — https://github.com/mit-han-lab/llm-awq
  - 4단계 구조(fake quant → real quant → pack&save → load&inference)를 코드와 플래그로 명확히 나눠놓은 레퍼런스 (커널은 W4A16).
  - `awq/quantize/quantizer.py::pseudo_quantize_tensor()` — fake quant
  - `--q_backend real` — real quant 전환
  - `awq/quantize/qmodule.py::WQLinear.from_linear()` — packing
  - `tinychat` / `awq_cuda` — load + CUDA 커널 추론
- **HandH1998/QQQ** — https://github.com/HandH1998/QQQ (W4A8, `--w_quantizer FixedQuantize`, `--w_group_size -1` = per-channel. per-channel W4A8 GEMM과 per-group W4A8 GEMM이 별도 구현. Ampere 이상, `pip install -e .` 빌드. vLLM에 통합됐다고 README에 명시)
- **IST-DASLab/marlin** — https://github.com/IST-DASLab/marlin (symmetric int4 커널)
- **mit-han-lab/omniserve (QServe)** — https://github.com/mit-han-lab/omniserve (W4A8KV4)
- **vllm-project/llm-compressor** — https://github.com/vllm-project/llm-compressor (프로젝트 구조 참고)
- vLLM W4A8 문서 — https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w4a8/

### 초기 4단계 파이프라인
```
1) fake quant RTN: group별 scale = max(|w_group|)/qmax, q = clamp(round(w/scale)), w_dequant = q*scale
2) real quant RTN: 같은 로직, int8 컨테이너에 int4 값(-8~7) 저장. fake quant와 수학적으로 동일해야 함
3) packing and save: packed = (high<<4) | low, high/low는 (q+8). scale·group_size·n_bits와 함께 저장
4) load and inference: activation은 per-token dynamic int8. group별 int32 누적 → group scale 곱 → 합산 → activation scale 곱
```

### 초기 numpy 검증 (합성 데이터)
- fake quant dequant 평균 절대오차 ~0.0045
- real quant와 fake quant 역산값 정확히 일치
- int4 pack/unpack round-trip 완전 일치
- naive W4A8 커널 시뮬레이션 vs fp32 행렬곱: 평균 상대오차 ~10.2% (합성 랜덤 데이터 기준)
- 검증 스크립트는 파일로 저장된 적 없음. 같은 로직을 C++로 옮긴 것이 `csrc/w4a8_rtn_naive.cu`.
