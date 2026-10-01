# MoE Router 통계 수집기 (`moe_routing_stats.py`)

MoE 모델의 **프리필 단계에서 layer별로 몇 개의 expert가 실제로 사용되는지**를 모델별, 도메인별, 입력 길이별로 측정합니다.
측정 결과는 NPU 성능 예측 엑셀(`LLM_NPU_성능예측.xlsx`)의 TTFT 계산에 반영할 수 있습니다.

---

## 1. 왜 측정하나

- 프리필 시간은 대부분 "N개 토큰이 사용하는 expert 가중치를 DRAM에서 읽는 시간"이 차지합니다. 그래서 **사용되는 서로 다른 expert 수**가 TTFT를 거의 결정합니다.
- 엑셀은 기본으로 **균등 라우팅**을 가정합니다.

  ```
  사용 비율 = 1 − (1 − k/E)^N        (E: expert 수, k: 토큰당 활성 expert 수, N: 입력 길이)
  사용 개수 = E × 사용 비율
  ```

  토큰 하나가 특정 expert를 고르지 않을 확률은 `1 − k/E`이고, 토큰끼리 독립이면 N개 토큰 모두가 그 expert를 고르지 않을 확률은 `(1 − k/E)^N`입니다.
- 균등 가정은 **보수적인 상한**입니다. 실제 라우팅은 쏠림이 있어서 사용 expert 수가 이 값보다 작거나 같습니다. 따라서 측정하면 TTFT 예측은 그대로이거나 줄어들 뿐, 늘어나지 않습니다.
- 균등 가정에서의 사용 비율 예시 (layer당)

  | 입력 길이 | gpt-oss-20b (E=32, k=4) | Qwen3.6-35B-A3B (E=256, k=8) | Gemma 4 26B-A4B (E=128, k=8) |
  |---|---|---|---|
  | 64 | 99.98% | 86.9% | 98.4% |
  | 128 | 100% | 98.3% | 99.97% |
  | 256 이상 | 100% | 99.97% 이상 | 100% |

  → 실측이 의미 있는 구간은 주로 **128~512 토큰의 짧은 입력**입니다.
- 디코드(batch 1)는 라우팅 분포와 관계없이 토큰마다 k개 expert를 읽습니다. 계산기는 expert 재사용이 없다고 가정하므로 실측 여부와 무관합니다.

---

## 2. 동작 방식

- 각 MoE layer의 router 모듈에 forward hook을 걸어, 토큰별로 선택된 top-k expert 인덱스를 기록합니다.
  - gpt-oss, Qwen3.6, Gemma 4 모두 router가 `(…, top_k_index)`를 마지막 출력으로 반환합니다 (transformers ≥ 5.5 기준).
  - router가 인덱스를 반환하지 않는 모델은 logits에서 직접 top-k를 구합니다.
- **최대 길이로 한 번만 forward합니다.** causal 모델은 앞쪽 L개 토큰의 라우팅이 뒤 토큰과 무관하므로, 8192 토큰을 한 번 돌리고 앞에서부터 128, 256, …, 8192 토큰 구간의 고유 expert 수를 세면 모든 길이의 결과를 한 번에 얻습니다.
- 한 샘플은 **같은 도메인의 문서만** 이어 붙여 만듭니다. 서로 다른 도메인을 한 프롬프트에 섞으면 라우팅이 인위적으로 넓게 퍼지기 때문입니다.

---

## 3. 설치

```bash
pip install torch "transformers>=5.5" datasets pandas matplotlib
```

- 모델이 올라갈 GPU 메모리가 필요합니다 (bf16 기준 Qwen3.6-35B-A3B 약 70GB, Gemma 4 26B-A4B 약 50GB, gpt-oss-20b 약 42GB). `--device-map auto`로 여러 GPU에 나눠 올릴 수 있습니다.
- gated 모델이나 데이터셋은 `huggingface-cli login`이 필요할 수 있습니다.

---

## 4. 실행

```bash
# 모델별 측정 (도메인당 32개 샘플, 기본 입력 길이 128 ~ 8192)
python moe_routing_stats.py --model openai/gpt-oss-20b      --out results --samples 32
python moe_routing_stats.py --model Qwen/Qwen3.6-35B-A3B    --out results --samples 32
python moe_routing_stats.py --model google/gemma-4-26b-a4b  --out results --samples 32
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--model` | HF 모델 id 또는 로컬 경로 | (필수) |
| `--out` | 결과 저장 폴더 | `routing_results` |
| `--lengths` | 측정할 입력 길이 목록 | `128 256 512 1024 2048 4096 8192` |
| `--samples` | 도메인당 샘플 수 | `32` |
| `--domains` | 기본 제공 도메인 중 사용할 것 | `chat code math wiki ko` |
| `--data 이름=경로` | 내 데이터를 도메인으로 추가 (`.jsonl`의 `text` 또는 `messages` 필드, 또는 `.txt` 한 줄 = 한 문서) | 없음 |
| `--dtype` | 모델 로드 정밀도 | `bfloat16` |
| `--device-map` | 모델 배치 | `auto` |
| `--synthetic` | 실제 데이터 대신 랜덤 토큰 사용 (hook 동작 점검용) | 꺼짐 |
| `--analyze 경로` | 수집된 `raw_*.csv`로 분석만 다시 실행 | 없음 |
| `--simulate` | torch 없이 균등/편향 라우팅 시뮬레이션 (파이프라인 검증용) | 꺼짐 |

### 기본 제공 도메인

| 이름 | 데이터셋 | 용도 |
|---|---|---|
| `chat` | HuggingFaceH4/ultrachat_200k (chat template 적용) | 일반 대화 |
| `code` | codeparrot/codeparrot-clean-valid | 코드 |
| `math` | openai/gsm8k | 수학 |
| `wiki` | Salesforce/wikitext (wikitext-103) | 일반 문서 |
| `ko` | wikimedia/wikipedia (20231101.ko) | 한국어 |

데이터셋 접근이 안 되면 `--domains`에서 빼고 `--data`로 직접 지정하세요.
**실제 서비스 프롬프트가 있다면 그것이 가장 좋은 대표 데이터입니다.** 예: `--data service=/data/prompts.jsonl`

### 권장 실행 순서

```bash
# 1) hook 동작 점검 (데이터 다운로드 없이 랜덤 토큰으로 빠르게)
python moe_routing_stats.py --model Qwen/Qwen3.6-35B-A3B --synthetic --samples 2 --domains chat --out check

# 2) 본 측정
python moe_routing_stats.py --model Qwen/Qwen3.6-35B-A3B --out results --samples 32
```

---

## 5. 출력 파일

| 파일 | 내용 |
|---|---|
| `raw_<모델>.csv` | 원자료: domain, sample, length, layer별 사용 expert 수 |
| `layer_stats_<모델>.csv` | domain × length × layer별 mean / p95 / max, 균등 가정 값 |
| `conservative_<모델>.csv` | length × layer 보수값 (도메인별 p95 중 최댓값) |
| `excel_paste_<모델>.csv` | length별 보수값의 layer 평균 → **엑셀에 붙여 넣을 값** |
| `eeff_<모델>.csv` | domain × layer별 유효 expert 수 `E_eff = exp(entropy)` |
| `heatmap_<모델>.png` | layer × length 사용 비율 히트맵 (보수값 기준) |

**보수값 정의:** 각 (입력 길이, layer)마다 도메인별로 샘플의 95번째 백분위수(p95)를 구하고, 그중 최댓값을 취합니다.

---

## 6. 결과 해석

### 6-1. `excel_paste_<모델>.csv` — 가장 먼저 볼 파일

입력 길이별로 `measured_ratio`(실측)와 `uniform_ratio`(균등 가정)를 비교합니다.

| 상황 | 의미 | 조치 |
|---|---|---|
| 두 값이 거의 같음 (예: 99% 대 100%) | 균등 가정이 현실과 맞음 | 엑셀 '균등' 모드 유지 |
| 실측이 눈에 띄게 낮음 (예: 78% 대 98%) | 라우팅이 쏠려 있어 짧은 입력의 TTFT가 실제로 더 짧음 | 엑셀에 실측값 입력 후 '실측' 모드 |
| 실측이 균등보다 높음 | 이론상 나올 수 없음 (균등이 상한) | 샘플 수 부족 또는 데이터 구성 문제 → 샘플을 늘려 재측정 |

입력이 길어질수록 두 값은 100%로 수렴합니다.

### 6-2. `heatmap_<모델>.png` — layer별 패턴

세로축은 layer, 가로축은 입력 길이, 색은 사용 비율입니다 (밝을수록 100%).

| 패턴 | 의미 |
|---|---|
| 전체가 고르게 밝음 | layer 간 차이가 없음. 엑셀처럼 layer 평균으로 충분 |
| 특정 layer만 어두움 | 그 layer는 소수 expert에 몰림 (흔히 앞쪽 또는 뒤쪽 layer). 성능 예측에는 평균으로 반영되며, 자주 쓰이는 expert를 SRAM에 미리 올려 두는 최적화를 검토할 만한 후보 |
| 짧은 입력 쪽만 어둡고 길어지면 밝아짐 | 정상 패턴 |

### 6-3. `layer_stats_<모델>.csv` — 도메인별 차이

같은 길이와 layer에서 도메인별 `mean`과 `p95`를 비교합니다.

| 상황 | 의미 |
|---|---|
| 도메인 간 차이가 작음 | 라우팅이 도메인을 크게 타지 않음. 어떤 데이터로 측정해도 결론이 같음 |
| 특정 도메인만 낮음 (예: code만 70%) | 그 도메인은 쓰는 expert가 한정적. 보수값(도메인 최댓값)에는 영향이 없지만, 그 도메인 위주의 서비스라면 TTFT가 더 짧다는 근거가 됨 |
| mean과 p95 차이가 큼 | 프롬프트마다 편차가 큼 → 샘플 수를 늘리는 것이 좋음 |

### 6-4. `eeff_<모델>.csv` — 쏠림 정도를 숫자 하나로

`E_eff ÷ E`는 라우팅이 "사실상 몇 %의 expert에 균등하게 퍼진 것과 같은지"를 나타냅니다.

- 0.9 이상: 거의 균등
- 0.5 정도: 절반의 expert에 몰린 것과 같음

이 값은 데이터셋 전체 평균이라 한 프롬프트 안의 쏠림은 반영하지 못합니다. **판단은 6-1 ~ 6-3으로 하고, E_eff는 모델끼리 쏠림 정도를 비교하는 참고용**으로 사용하세요.

### 6-5. 결론 내리는 기준

- 128 토큰 기준으로 `measured_ratio`가 `uniform_ratio`보다 **5%p 이상 낮으면** 실측 모드를 쓸 가치가 있습니다.
- 그보다 차이가 작으면 균등 가정을 유지하는 것이 단순하고 보수적입니다.
- 모델마다 판단이 다를 수 있습니다. Qwen3.6처럼 expert가 많을수록(256개) 차이가 날 여지가 크고, gpt-oss처럼 적을수록(32개) 거의 차이가 없습니다.

---

## 7. 엑셀 계산기에 반영하기

1. `excel_paste_<모델>.csv`의 `measured_used_layer_avg` 열을 `LLM_NPU_성능예측.xlsx`의 **'Expert 실측'** 시트 ①번 표에서 해당 모델 칸에 붙여 넣습니다. 입력 길이 순서가 같아야 합니다.
2. **'설정'** 시트의 **"프리필 expert 사용 수 기준"**을 `균등`에서 `실측`으로 바꿉니다.
3. 'Expert 실측' 시트 ③번 표에서 실제 계산에 적용되는 비율을 확인합니다. 비워 둔 칸은 균등 공식으로 계산됩니다.

엑셀에는 layer 평균값을 넣습니다. 가중치를 읽는 총량은 layer별 사용 수의 합에 비례하므로, 평균을 넣어도 총량은 정확합니다. layer별 차이는 히트맵과 CSV에서 확인하세요.

---

## 8. 검증과 한계

- **검증 완료:** `--simulate`로 분석 파이프라인을 검증했습니다.
  - 균등 라우팅 시뮬레이션의 평균 사용 수는 251.3개로, 엑셀 공식 값 251.6개와 일치했습니다 (E=256, k=8, 128 토큰).
  - 쏠린 분포에서는 203개로 더 낮게 나와, 균등 가정이 상한임을 확인했습니다.
- **미검증:** 실제 모델에서의 hook 부분은 GPU가 없는 환경에서 작성되어 실행해 보지 못했습니다. 본 측정 전에 `--synthetic`으로 먼저 점검하세요.
- **알려진 한계**
  - batch 1, 프리필 기준입니다.
  - 샘플은 같은 도메인 문서를 이어 붙여 만들기 때문에, 긴 길이(4K~8K)에서는 문서 경계가 포함됩니다.
  - 경량화 전(bf16) 모델로 측정하면 경량화 후 라우팅과 약간 다를 수 있습니다. 차이가 걱정되면 경량화된 모델로 측정하세요.
  - vision/audio tower와 Qwen3.6의 MTP layer는 측정 대상이 아닙니다.
