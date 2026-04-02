# 세션 대화 기록

작성일: 2026-03-24

---

## 프로젝트 개요

- **과제**: DialogSum-KO 데이터셋 기반 대화 요약 NLP 경진대회
- **평가 지표**: ROUGE-1 + ROUGE-2 + ROUGE-L F1 합산 (MeCab 형태소 분석)
- **실험 모델**: KoBART, mT5-base, SOLAR-10.7B-Instruct-v1.0

---

## 주요 작업 내역

### 1. 환경 구성 및 코드 생성

**요청**: `upstage/SOLAR-10.7B-Instruct` 및 `google/mt5-base` 모델 트레이닝 설정 생성

**생성/수정 파일**:
- `configs/solar_config.yaml` — SOLAR QLoRA 학습 설정
- `configs/mt5_config.yaml` — mT5-base Seq2Seq 학습 설정
- `src/training/seq2seq_trainer.py` — MT5Trainer 추가, 버그 수정
- `src/training/trainer.py` — unsloth 제거 → 표준 PEFT+bitsandbytes로 재작성
- `train.py` — CLI 진입점 (kobart/mt5/solar 선택)
- `notebooks/03_train_solar.ipynb`
- `notebooks/03_train_mt5.ipynb`

---

### 2. KoBART 실험

#### 실험 1: 기본 설정 (lr=3e-5, ep=10)
- run_name: `kobart-lr3e5-ep10`
- ROUGE-1: ~0.23 수준 (Seq2Seq 한계)

#### 실험 2: lr 낮추기 + Early Stopping (lr=1e-5, patience=3)
- run_name: `kobart-lr1e5-ep20-stop`
- Early Stopping: epoch 14에서 조기 종료
- MeCab Score: **0.6000** (rouge1+rouge2+rougeL 합산)
- 제출 파일: `outputs/predictions/kobart-lr1e5-ep20-stop.csv`

**버그 수정**:
- `seq2seq_trainer.py`: `generate()` 호출 시 `token_type_ids` 제거 (KoBART 미지원)
- `EarlyStoppingCallback` 추가, `load_best_model_at_end=True` 설정

**리더보드 분석**:
- KoBART ROUGE-1 ~0.23 vs 상위팀 ~0.56 → 상위팀은 LLM+QLoRA 사용 확인
- 이에 따라 SOLAR-10.7B-Instruct QLoRA 실험으로 전환

---

### 3. 환경 문제 해결

#### 문제 1: unsloth 의존성 충돌
- `pip install unsloth_zoo`가 torch 2.11.0 (CUDA 13.0) 설치
- 시스템 드라이버는 CUDA 12.2 → GPU 인식 불가
- **해결**: torch 2.1.0+cu121 재설치, unsloth 완전 제거
  ```bash
  pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 --no-deps
  ```

#### 문제 2: bitsandbytes libcusparse.so.12 미발견
- **해결**: `pip install nvidia-cusparse-cu12` 설치 후 LD_LIBRARY_PATH 설정
  ```bash
  export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cusparse/lib:/opt/conda/lib:$LD_LIBRARY_PATH
  ```

#### 문제 3: SOLAR 모델 401 Unauthorized
- `upstage/SOLAR-10.7B-Instruct`: gated 모델 (HuggingFace 인증 필요)
- **해결**: `huggingface-cli login --token hf_...` (토큰: nlp-jacky로 저장)

#### 문제 4: SOLAR 모델명 404 Not Found
- `upstage/SOLAR-10.7B-Instruct` → 404
- **해결**: `configs/solar_config.yaml`의 모델명을 `upstage/SOLAR-10.7B-Instruct-v1.0`으로 수정

#### 문제 5: bitsandbytes optimizer TypeError
- `learning_rate: 2e-4` (YAML) → 문자열로 파싱됨
- **해결**: `configs/solar_config.yaml`에서 `learning_rate: 0.0002`로 변경

---

### 4. trainer.py 재작성 (unsloth → 표준 PEFT)

**주요 변경 사항**:

| 항목 | 이전 (unsloth) | 이후 (표준) |
|---|---|---|
| 모델 로드 | `FastLanguageModel.from_pretrained` | `AutoModelForCausalLM.from_pretrained` + `BitsAndBytesConfig` |
| LoRA 적용 | `FastLanguageModel.get_peft_model` | `prepare_model_for_kbit_training` + `get_peft_model` |
| 추론 모드 | `FastLanguageModel.for_inference` | `model.eval()` |
| Response-only | unsloth 내장 | 커스텀 `ResponseOnlyCollator` 구현 |

`_apply_response_only()`: `### Assistant:\n` 토큰 이전 구간에 label=-100 마스킹

---

### 5. SOLAR 학습 실행 (현재 진행 중)

**실행 명령**:
```bash
export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cusparse/lib:/opt/conda/lib:$LD_LIBRARY_PATH
python train.py --model solar --run-name solar-r16-lr2e4-ep3 --no-notion
```

**설정**:
| 항목 | 값 |
|---|---|
| 모델 | upstage/SOLAR-10.7B-Instruct-v1.0 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Learning rate | 0.0002 (2e-4) |
| Epochs | 3 |
| Batch (실효) | 1 × 16 = 16 |
| 4-bit QLoRA | True |
| BF16 | True |
| Response-only | True (### Assistant: 마커) |
| 총 파라미터 | 5.5B |
| 학습 파라미터 | 62.9M (1.13%) |
| 총 steps | 2,334 |

**진행 현황** (확인 시점 기준):
- step/total: 15/2334 (0.6%)
- 속도: ~24초/step
- 예상 소요: ~15.6시간
- WandB: https://wandb.ai/sskim7415-/dialogue-summarization/runs/1dc6uzzk
- Notion 페이지 ID: 32d5930a-a61a-8107-9862-dd685a800f59 (수동 등록)
- Experiment ID: EXP-20260324-111751

**프롬프트 형식**:
```
### User:
다음 대화를 한국어로 간결하게 요약하세요.

{dialogue}

### Assistant:
{summary}
```

---

## 향후 계획

1. SOLAR 학습 완료 후 dev ROUGE 점수 확인 및 Notion 업데이트
2. test set 추론 → 제출 파일 생성
3. 필요 시 추가 실험 (LoRA rank 확대, prompt 개선 등)

---

## [2026-03-24~25] SOLAR ep2 결과 및 프롬프트 문제 발견

### SOLAR ep2 (solar-r16-lr2e4-ep2) 결과

- 체크포인트: checkpoint-1557 (epoch 2 / 2334 steps)
- Dev ROUGE-1: **0.1372** / Score: 0.3208
- **문제**: 요약이 영어로 생성되는 비율이 높음
  - 예: `"Person1은 Person2에게 internal memo를 all staff에게..."`
- **원인**: 프롬프트에 한국어 출력 지시 부족 + SOLAR의 영어 생성 편향

### 프롬프트 개선 (2026-03-26)

기존:
```
다음 대화를 한국어로 간결하게 요약하세요.
```

개선:
```
당신은 한국어 대화 요약 전문가입니다. 대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다.
요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요.
핵심 내용만 1~3문장으로 간결하게 요약하세요.

아래 대화를 읽고 핵심 내용을 한국어로 요약해주세요. 화자 태그(#Person1# 등)를 유지하세요.
```

---

## [2026-03-26] ep1-newprompt 학습 및 추론 트러블슈팅

### SOLAR ep1-newprompt 학습

- run_name: `solar-r16-lr2e4-ep1-newprompt`
- 실행: 2026-03-26 05:08 ~ 11:29 (6시간 21분)
- 총 steps: 778 (epoch 1), 속도 ~29초/step
- 체크포인트: `outputs/checkpoints/EXP-20260326-050841/checkpoint-778`

### 추론 단계 트러블슈팅 (6가지 문제)

**문제 1. 로그 11:29 이후 미업데이트**
- 원인: Python stdout full buffering (tqdm만 stderr → 즉시 기록, print()는 버퍼 대기)
- 확인: `/proc/PID/fdinfo/1`의 pos 값 = 파일 크기 → flush 없음
- 해결: `python -u` + `print(flush=True)`

**문제 2. 추론 속도 490초/샘플 (unsloth 없을 때)**
- 원인: 4-bit QLoRA 추론 시 매 스텝 NF4 역양자화로 메모리 대역폭 병목
- 영향: dev+test 추론에 ~136시간 예상
- 해결: unsloth 설치 + `FastLanguageModel.for_inference()` → **50초/샘플 (10배 개선)**

**문제 3. unsloth 설치로 패키지 대규모 버전 업**

| 패키지 | 이전 | 이후 |
|--------|------|------|
| torch | 2.1.0+cu121 | 2.10.0 |
| transformers | 4.44.0 | 5.3.0 |
| trl | 0.9.6 | 0.24.0 |
| peft | 0.12.0 | 0.18.1 |

- 주의: 재학습 시 `SFTTrainer` 구버전 API → `SFTConfig` 방식으로 수정 필요

**문제 4. `No module named 'unsloth'`**
- train.py 제출 루프에 unsloth import가 있었으나 당시 미설치
- 해결: 전용 `infer_solar_unsloth.py` 스크립트 분리

**문제 5. `RuntimeError: Failed to find C compiler`**
- triton 3.6.0이 gcc 필요, 미설치 상태
- 해결: `apt-get install -y gcc`

**문제 6. GPU OOM (이전 프로세스 잔재)**
- kill 후에도 자식 프로세스가 ~8GB × 3개 GPU 점유
- 해결: `ps aux | grep python`으로 전체 확인 후 kill

### 추론 결과 (infer_solar_unsloth.py 완료)

- Dev ROUGE-1: 0.1570 / ROUGE-2: 0.1169 / ROUGE-L: 0.1492 / Score: 0.4231
- 제출 파일 생성: `outputs/predictions/solar-r16-lr2e4-ep1-newprompt.csv` (499개)

---

## [2026-03-27] 제출 파일 다국어 출력 후처리

### 문제 발견

499개 제출 파일 분석 결과:
- **309개 (61.9%)**: 한글 비율 30% 미만 (비정상 출력)
  - 순수 영어: 136개 (27.3%)
  - 조사만 한글, 본문은 러시아어·스페인어 등 혼합: 추가 다수
  - 예시(test_19): `#Person1#은 #Person2#를 помочь просит. #Person2# хочет купить новый телефон...`
  - 예시(test_6): `#Person1#은 #Person2#의불편함과 scratching에해perturbadoestá...`

### 원인

- SOLAR-10.7B는 다국어 사전학습 모델. 1 epoch 파인튜닝으로는 비한국어 생성 편향 미극복
- 단순 `is_korean()` 체크(한글 1자 이상)로는 혼합 케이스 미감지

### 조치

- **감지 로직 개선**: 한글 비율 30% 미만 → 재추론 대상
- **강화 프롬프트 적용**:
  `[중요] 반드시 한국어로만 작성하세요. 영어 사용 금지. MUST respond in Korean only.`
- **스크립트**: `fix_english_outputs.py`
- 원본 백업: `solar-r16-lr2e4-ep1-newprompt_before_fix.csv`

### 현재 상태 (2026-03-27)

- `fix_english_outputs.py` PID 671891 실행 중
- 재추론 대상: **309개**, 속도 ~50초/샘플, 예상 완료 ~4시간 20분
- 완료 후 제출 파일 자동 업데이트 (백업 보존)

---

## 주요 파일 경로

```
dialogue-summarization/
├── configs/
│   ├── solar_config.yaml       # SOLAR QLoRA 설정
│   ├── kobart_config.yaml      # KoBART 설정
│   └── mt5_config.yaml         # mT5 설정
├── src/
│   ├── training/
│   │   ├── trainer.py          # SOLAR/LLM 트레이너 (PEFT+bitsandbytes)
│   │   └── seq2seq_trainer.py  # KoBART/mT5 트레이너
│   └── utils/
│       ├── notion_logger.py
│       └── wandb_logger.py
├── outputs/
│   ├── checkpoints/            # 모델 체크포인트
│   └── predictions/            # 제출용 CSV
└── train.py                    # CLI 진입점
```

---

## 환경 정보

- GPU: RTX 3090 (24GB)
- CUDA Driver: 12.2 (535.86.10)

| 패키지 | 초기 버전 | 현재 버전 (2026-03-27) |
|--------|----------|----------------------|
| torch | 2.1.0+cu121 | 2.10.0 |
| transformers | 4.44.0 | 5.3.0 |
| trl | 0.9.6 | 0.24.0 |
| peft | 0.12.0 | 0.18.1 |
| bitsandbytes | 0.41.3 | 0.49.2 |
| unsloth | 미설치 | 2026.3.15 (2026-03-26 설치) |
