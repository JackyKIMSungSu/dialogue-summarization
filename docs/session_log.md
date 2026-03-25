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

- GPU: RTX 3090
- CUDA Driver: 12.2 (535.86.10)
- torch: 2.1.0+cu121
- transformers: 4.44.0
- peft: (최신)
- bitsandbytes: (최신)
- trl: 0.9.6
