# Dialogue Summarization

일상 대화를 요약하는 모델을 개발하는 NLP 경진대회 프로젝트입니다.
DialogSum-KO 데이터셋 기반, ROUGE-1 + ROUGE-2 + ROUGE-L F1 합산을 최종 점수로 평가합니다.

---

## 프로젝트 구조

```
dialogue-summarization/
├── configs/
│   ├── base_config.yaml          # LLM (Qwen3 등) 기본 설정
│   └── kobart_config.yaml        # KoBART (Seq2Seq) 설정
│
├── data/
│   ├── raw/                      # 원본 데이터 (train/dev/test/sample_submission.csv)
│   ├── processed/                # 전처리 완료 데이터
│   └── external/                 # 허용된 외부 데이터
│
├── notebooks/
│   ├── 01_data_analysis.ipynb    # 데이터 탐색 및 통계 분석
│   ├── 02_preprocessing.ipynb    # 전처리 실행 및 저장
│   ├── 03_train.ipynb            # LLM (Unsloth/QLoRA) 학습
│   └── 03_train_kobart.ipynb     # KoBART (Seq2Seq) 학습
│
├── src/
│   ├── data/
│   │   ├── preprocessor.py       # DialoguePreprocessor
│   │   ├── dataset.py            # DialogueSummarizationDataset
│   │   └── data_loader.py        # build_dataloaders()
│   ├── training/
│   │   ├── trainer.py            # LLM 트레이너 (HyperParams, DialogueSummarizationTrainer)
│   │   └── seq2seq_trainer.py    # KoBART 트레이너 (Seq2SeqHyperParams, KoBARTTrainer)
│   ├── inference/
│   │   └── submit.py             # SubmissionGenerator
│   └── utils/
│       ├── rouge_evaluator.py    # RougeEvaluator (MeCab 기반 공식 평가)
│       ├── wandb_logger.py       # WandbLogger
│       └── notion_logger.py      # NotionLogger (Training Datasheet 연동)
│
├── outputs/
│   ├── checkpoints/              # 모델 체크포인트
│   ├── predictions/              # 제출용 CSV
│   └── logs/                     # 학습 로그
│
├── .env.template                 # 환경변수 템플릿
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 데이터셋

| 분할 | 샘플 수 | 컬럼 |
|---|---|---|
| train | 12,457 | fname, dialogue, summary, topic |
| dev | 499 | fname, dialogue, summary, topic |
| test | 499 | fname, dialogue (summary 없음) |

**주요 통계 (train)**

| 항목 | 평균 | 최대 |
|---|---|---|
| 대화 길이 (문자) | 439 | 2,546 |
| 요약 길이 (문자) | 87 | 478 |
| 대화 턴 수 | 9.5 | 61 |

---

## 평가 지표

대회 공식 평가: **MeCab 형태소 분석 후 ROUGE F1 합산**

```
최종 점수 = ROUGE-1-F1 + ROUGE-2-F1 + ROUGE-L-F1
```

| 기준 | 점수 |
|---|---|
| 정답 3개 중 랜덤 1개 선택 시 | ~70점 |
| Baseline SFT (Standard) | ~0.32 (ROUGE-1) |
| Baseline SFT (Response-Only) | ~0.56 (ROUGE-1) |

---

## 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.template .env
# .env 파일에 아래 값 입력:
#   WANDB_API_KEY=
#   NOTION_TOKEN=
#   NOTION_DATABASE_ID=
```

---

## 실행 순서

### 1단계: 데이터 분석
`notebooks/01_data_analysis.ipynb` 실행

### 2단계: 전처리
`notebooks/02_preprocessing.ipynb` 실행
→ `data/processed/` 에 전처리 데이터 저장

### 3단계: 학습

**KoBART (빠른 실험)**
```python
# notebooks/03_train_kobart.ipynb

hp = Seq2SeqHyperParams(
    train=Seq2SeqTrainConfig(learning_rate=3e-5, num_epochs=10),
    generation=GenerationConfig(num_beams=4),
    experiment=Seq2SeqExperimentConfig(run_name="kobart-exp01"),
)
trainer = KoBARTTrainer(hp, notion_logger=notion, wandb_logger=wandb)
trainer.run(train_df, dev_df)
```

**LLM + QLoRA (고성능)**
```python
# notebooks/03_train.ipynb

hp = HyperParams(
    model=ModelConfig(model_name="unsloth/Qwen3-14B"),
    lora=LoraConfig(r=16, alpha=16),
    train=TrainConfig(learning_rate=2e-4, num_epochs=3),
    experiment=ExperimentConfig(run_name="qwen3-exp01"),
)
trainer = DialogueSummarizationTrainer(hp, notion_logger=notion, wandb_logger=wandb)
trainer.run(train_df, dev_df)
```

### 4단계: 제출 파일 생성
```python
from src.inference.submit import SubmissionGenerator

gen = SubmissionGenerator()
gen.save(predictions=predictions, filename="exp01.csv")
```

---

## 모델 선택 가이드

이 프로젝트에서 학습하는 3가지 모델:

| 모델 | 타입 | VRAM | 특징 | Config | Notebook |
|---|---|---|---|---|---|
| `gogamza/kobart-base-v2` | Seq2Seq | ~4GB | 경량, 빠른 실험, 한국어 특화 | `kobart_config.yaml` | `03_train_kobart.ipynb` |
| `google/mt5-base` | Seq2Seq | ~4GB | 다국어 T5, 한국어 지원 | `mt5_config.yaml` | `03_train_mt5.ipynb` |
| `upstage/SOLAR-10.7B-Instruct` | Causal LM | ~14GB | 고성능 LLM, QLoRA 파인튜닝 | `solar_config.yaml` | `03_train_solar.ipynb` |

### CLI로 학습 실행

```bash
# KoBART
python train.py --model kobart --run-name kobart-exp01

# mt5-base
python train.py --model mt5 --run-name mt5-exp01

# SOLAR-10.7B-Instruct
python train.py --model solar --run-name solar-exp01

# 하이퍼파라미터 오버라이드
python train.py --model kobart --run-name kobart-lr1e5 --lr 1e-5 --epochs 5
```

---

## 실험 관리

- **WandB**: 학습 loss, ROUGE 점수 실시간 기록
- **Notion**: [Training Datasheet](https://www.notion.so/f7680aa2000a4d60b71b0606c5a0f9a6) — 실험 시작/종료/실패 자동 기록

학습 시작 시 Notion에 자동으로 행이 생성되며, 완료 후 ROUGE 점수가 업데이트됩니다.

---

## 대회 규칙 요약

- 일일 제출 횟수: **팀 단위 12회**
- DialogSum 데이터셋 직/간접 사용 **금지**
- 평가 데이터 학습 활용 **금지**
- 무료 API만 허용 (Solar 모델 예외)
- 최종 제출: 팀/개인별 **2개**
