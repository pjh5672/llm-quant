# W4A8 Symmetric RTN Quantization — 작업 정리

## ▶ 이어서 하기 (마지막 업데이트: 2026-09-18)

### 현재 위치
- **Phase 0 완료. Phase 1 완료** (g128 재측정 끝, 아래 "Phase 1 결과"). **Phase 2는 아직 시작 전.**
- **granularity = group_size 128** (weight, activation 둘 다). "group 128 설계" 참고.
- **CUDA quant-dequant 커널 완성.** PyTorch 레퍼런스와 bit-exact. "bit-exact 규칙" 참고.
- **config 레이어 완성** (gaia-compressor 매핑). `configs/*.yaml` → `auto_llm.py` / `scheme_sweep.py`.
  sweep과 단일 실행이 `entrypoints/run.py::run_one` 같은 경로를 탐. 재현 확인 완료
  (W8A16 단일 실행 PPL이 sweep 값과 **완전히 동일**: 13.178328514099121).
- 테스트 **102개** 통과. `git init`만 하고 커밋은 아직 없음.

### 바로 다음에 할 일
Phase 1은 **다 끝났다**(9조합 + attn/mlp 16조합 + GEMM 속도 측정).
아래 "결정할 것" 2번만 정하면 Phase 2 진입. **A8/A16 선택이 Phase 2~3 작업을 거의 바꾸지 않으므로,
결정을 Phase 4로 미루고 먼저 진행해도 된다.**

### 결정할 것 (Phase 2 시작 전)
1. **W4를 어떻게 할지.** ✅ **측정으로 결론남: 보완책 없이는 int4를 어디에도 못 쓴다.**
   - 전부 int4: +29.31% / mlp만 int4: +18.52% / attn만 int4: +5.95%. **전부 5% 기준 밖.**
   - 가장 아까운 건 attn만 int4(+5.95%, 1.346GB)지만 크기 이득이 5.5%뿐이라 교환이 나쁨.
   - 남은 선택지:
     - (a) **W8로 확정** → Phase 2 바로 진행 (권장, 가장 빠름)
     - (b) W4 **보완책** 구현 후 재측정: 채널별 clipping ratio 탐색(MSE 최소화) 또는 Hadamard rotation.
       **보완책은 MLP를 겨냥해야 한다** — 손실의 대부분이 MLP에서 나온다.
2. **W8A8과 W8A16 중 무엇으로 갈지.** ⬅ **지금 막혀 있는 유일한 결정**
   - A8 비용: PPL **+0.10%p**, 크기 이득 **0** (activation scale은 저장하지 않음).
   - A8 이득: int8 GEMM 속도뿐인데, 측정해보니 **prefill 1.03~1.04x, decode 0**
     (위 "속도 측정" 절). 기대했던 2x가 아니다.
   - **→ 현재 증거로는 W8A16이 낫다.** A8을 고르려면 Phase 4 커널이 `torch._int_mm`보다
     확실히 빠르다는 걸 보여야 한다. 입증 책임이 A8 쪽에 있다.
   - 선택지: (a) **W8A16으로 확정**하고 Phase 2 진행 (권장),
     (b) Phase 4에서 커널 속도를 본 뒤 되돌아와 결정 (A8 가능성을 열어둠).
     어느 쪽이든 Phase 2~3 작업 내용은 **거의 같다** — real quant와 packing은 weight 기준이고,
     activation은 저장할 scale이 없어서 config 한 줄 차이다.
3. scale dtype fp32 유지 여부. g128이라 scale이 약 1.5MB → 약 30MB로 늘었음. 지금은 fp32 유지.
4. head_weight는 **bf16으로 확정해도 됨.** 3번의 독립 측정에서 일관되게 크기(-252MB)도 PPL도 더 나음.

### 끝까지 확인하지 못한 것
- 없음. 이전의 "csrc가 컴파일된 적 없음"은 해결됨 ("Windows 빌드 환경" 참고).
- `csrc/w4a8_rtn_naive.cu`, `build_and_run.bat`은 초기 group-wise 설계의 잔재. 현재 커널
  (`src/llmquant/kernels/`)과 무관하므로 지워도 됨.

### 다시 시작하는 방법
```powershell
cd C:\Users\Park Jiho\Desktop\Project\DEV\llm-quant
.\.venv\Scripts\python.exe -m pytest tests -q                                      # 84개
.\.venv\Scripts\python.exe examples\auto_llm.py --cfg configs\llama3.2-1b-w8a16.yaml   # 단일 실행 ~1분
.\.venv\Scripts\python.exe examples\scheme_sweep.py --cfg configs\sweep-phase1.yaml    # 16런 ~18분
```
- 패키지 재설치: `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` (ninja 포함)
- PowerShell에서 `$env:PYTHONIOENCODING="utf-8"` 권장.
- CUDA 커널은 첫 호출 때 자동 JIT 빌드(1~2분), 이후 캐시.
- 결과는 `experiments/<project>/`에 저장됨 (config 복사 + `result.json` / `sweep.json`).

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
- **모든 Linear의 in_features는 2048 또는 8192** (k/v_proj도 in=2048)이라 128로 나누어떨어짐. 패딩 불필요.

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

| 2 | **real quant** reference: packing 안 한 int weight + 검증된 PyTorch 연산(A8: `torch._int_mm`, A16: dequant→bf16 matmul)으로 PPL 측정 | fake와 real의 PPL 차이가 작음 (크면 버그) |
| 3 | **packing → bin 파일 저장 → Python 로드** | 로드한 정수와 scale이 Phase 2와 bit-exact |
| 4 | **custom CUDA 커널**이 packing 데이터를 직접 읽어서 연산 | op별 출력이 Phase 2와 bit-exact, 전체 PPL 완전 동일, 속도·VRAM 측정 |
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
  `mode=real/kernel`, `quant_method != rtn`, `kv_cache != bf16`). "설정했는데 아무 일도
  안 일어나는" 상태를 만들지 않기 위함.

### kv_cache가 미구현인 이유
1. Linear 경로가 아니라 **attention 경로**를 건드림. transformers `Cache`를 상속해 K/V 저장을
   가로채야 하고, "양자화하지 않는다"고 정한 attention BMM에 영향이 감.
2. **현재 PPL 지표로는 효과가 0.** `evaluate_ppl`은 2048 토큰 청크마다 `model(batch)`를
   한 번씩만 돌려서 캐시가 쓰이기만 하고 **다음 스텝에서 읽히지 않음.** K/V를 아무리
   양자화해도 PPL이 그대로임. 효과를 보려면 생성 기반 지표(Phase 5)가 따로 필요함.

→ 스키마 고정을 위해 슬롯만 두고, 구현과 전용 pass 조건은 Phase 5 근처로 미룸.

## 프로젝트 구조

llm-compressor 구조 + gaia-compressor의 config/파이프라인 레이어. `src/llmquant/` 패키지(`pip install -e ".[dev]"`).

```
llm-quant/
├── pyproject.toml, .gitignore
├── docs/w4a8_rtn_notes.md               # 이 파일
├── configs/*.yaml                       # 실행 config (llama3.2-1b-w4a8, w8a16, bf16, sweep-phase1)
├── src/llmquant/
│   ├── __init__.py                      # oneshot, QuantizationModifier, evaluate_ppl, generate
│   ├── args/__init__.py                 # ModelArgs, DatasetArgs
│   ├── args/quant_config.py             # QuantConfig(attn/mlp/head/activation/kv_cache/group_size)
│   ├── args/parser.py                   # RunConfig, build_config(YAML+CLI), expand_sweep
│   ├── datasets/wikitext.py             # get_eval_ids("wikitext2", tokenizer)
│   ├── entrypoints/oneshot.py           # oneshot(model, recipe)
│   ├── entrypoints/evaluate.py          # evaluate_ppl, generate, CLI(bf16 기준)
│   ├── entrypoints/run.py               # run_one(config) — 단일 실행과 sweep의 공통 경로
│   ├── analysis.py                      # sweep 결과 분석 (순수 함수, GPU/모델 불필요)
│   ├── benchmark.py                     # GEMM 연산 벤치 (bf16 vs int8 vs int8 g128)
│   ├── modifiers/quantization/scheme.py # QuantizationArgs, QuantizationScheme, PRESET_SCHEMES
│   ├── modifiers/quantization/modifier.py # QuantizationModifier(scheme, attn_scheme, mlp_scheme, lm_head_scheme, mode)
│   ├── observers/minmax.py              # compute_scale: channel(weight) / token(activation), fp32
│   ├── modules/fake_quant_linear.py     # FakeQuantLinear
│   └── utils/quant_ops.py, model.py, size.py
├── src/llmquant/kernels/                # CUDA 커널 (JIT 빌드)
│   ├── build.py                         # MSVC env + TMP/8.3 + ninja 설정, load_extension()
│   ├── ops.py                           # fake_quantize_cuda(x, args, return_scale=False)
│   └── csrc/fake_quant.cu, fake_quant.cpp
├── examples/auto_llm.py                 # config 1개 실행
├── examples/scheme_sweep.py             # Phase 1, sweep 그리드 확장 + 분석 리포트
├── examples/analyze_sweep.py            # 저장된 sweep.json 재분석 (GPU 불필요)
├── examples/bench_gemm.py               # GEMM 속도 측정
├── tests/test_quantization.py           # 11개
├── tests/test_cuda_kernel.py            # 41개 (bit-exact 검증)
├── tests/test_config.py                 # 21개 (타깃 라우팅, bf16 규칙, activation 제한, kv_cache 슬롯)
├── tests/test_parser.py                 # 16개 (YAML/CLI 우선순위, sweep 확장)
├── tests/test_analysis.py               # 9개 (손으로 계산한 값과 대조)
├── tests/test_benchmark.py              # 4개
├── csrc/w4a8_rtn_naive.cu, build_and_run.bat   # 초기 naive 커널 초안 (컴파일 안 됨)
├── results/                             # git 제외 (구 결과 보관)
└── experiments/<project>/               # git 제외, config 복사 + result.json / sweep.json
```

사용 예시:
```bash
python examples/auto_llm.py --cfg configs/llama3.2-1b-w4a8.yaml
python examples/auto_llm.py --cfg configs/llama3.2-1b-w4a8.yaml --mlp-weight int8   # CLI가 이김
python examples/scheme_sweep.py --cfg configs/sweep-phase1.yaml
```
```python
from llmquant.args import QuantConfig
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
  구 sweep은 이걸 표현할 수 없었고, `configs/sweep-phase1.yaml`(16런)이 바로 이 영역을 덮는다.

## sweep 분석 (`llmquant/analysis.py`)

sweep 결과를 받아 **bit 조합 결정에 필요한 비교**를 자동으로 만들어 낸다. 전부 순수 함수라
저장된 `sweep.json`만 있으면 GPU도 모델도 없이 다시 돌릴 수 있다:

```bash
python examples/analyze_sweep.py experiments/phase1-sweep/sweep.json --limit-ratio 1.05
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
python examples/bench_gemm.py --m 1 64 512 2048
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
