"""
KoBART 등 Seq2Seq 모델 학습 트레이너.

HuggingFace Seq2SeqTrainer 기반.
WandB / Notion 연동 포함.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset


# ------------------------------------------------------------------
# 하이퍼파라미터 설정
# ------------------------------------------------------------------

@dataclass
class Seq2SeqModelConfig:
    model_name: str = "gogamza/kobart-base-v2"
    max_input_length: int = 1024
    max_output_length: int = 256


@dataclass
class Seq2SeqTrainConfig:
    num_epochs: int = 10
    batch_size: int = 16              # per_device_train_batch_size
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    grad_accum: int = 1
    fp16: bool = True
    label_smoothing: float = 0.1
    seed: int = 42
    early_stopping_patience: int = 0   # 0이면 비활성화, >0이면 Early Stopping 적용


@dataclass
class GenerationConfig:
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 256


@dataclass
class Seq2SeqExperimentConfig:
    experiment_id: str = ""
    run_name: str = ""
    dataset: str = "DialogSum-KO"
    purpose: str = ""
    tags: list[str] = field(default_factory=lambda: ["KoBART", "seq2seq"])
    output_dir: str = "outputs/checkpoints"
    prediction_dir: str = "outputs/predictions"

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = datetime.now().strftime("EXP-%Y%m%d-%H%M%S")
        if not self.run_name:
            self.run_name = self.experiment_id


@dataclass
class Seq2SeqHyperParams:
    model: Seq2SeqModelConfig = field(default_factory=Seq2SeqModelConfig)
    train: Seq2SeqTrainConfig = field(default_factory=Seq2SeqTrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    experiment: Seq2SeqExperimentConfig = field(default_factory=Seq2SeqExperimentConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Seq2SeqHyperParams":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            model=Seq2SeqModelConfig(
                model_name=cfg["model"]["name"],
                max_input_length=cfg["model"]["max_input_length"],
                max_output_length=cfg["model"]["max_output_length"],
            ),
            train=Seq2SeqTrainConfig(
                num_epochs=cfg["training"]["num_epochs"],
                batch_size=cfg["training"]["batch_size"],
                eval_batch_size=cfg["training"]["eval_batch_size"],
                learning_rate=cfg["training"]["learning_rate"],
                warmup_ratio=cfg["training"]["warmup_ratio"],
                weight_decay=cfg["training"]["weight_decay"],
                grad_accum=cfg["training"]["gradient_accumulation_steps"],
                fp16=cfg["training"]["fp16"],
                label_smoothing=cfg["training"].get("label_smoothing", 0.1),
                seed=cfg["training"]["seed"],
            ),
            generation=GenerationConfig(
                num_beams=cfg["generation"]["num_beams"],
                length_penalty=cfg["generation"]["length_penalty"],
                no_repeat_ngram_size=cfg["generation"]["no_repeat_ngram_size"],
                max_new_tokens=cfg["generation"]["max_new_tokens"],
            ),
        )

    def to_dict(self) -> dict:
        return {
            "model":      asdict(self.model),
            "train":      asdict(self.train),
            "generation": asdict(self.generation),
            "experiment": asdict(self.experiment),
        }

    def summary(self):
        t = self.train
        print("=" * 55)
        print(f"  실험 ID   : {self.experiment.experiment_id}")
        print(f"  Run Name  : {self.experiment.run_name}")
        print(f"  모델      : {self.model.model_name}")
        print(f"  LR        : {t.learning_rate}  |  Epochs: {t.num_epochs}")
        print(f"  Batch     : {t.batch_size} × {t.grad_accum} = {t.batch_size * t.grad_accum}")
        print(f"  Beams     : {self.generation.num_beams}  |  FP16: {t.fp16}")
        print(f"  Label smoothing: {t.label_smoothing}")
        print("=" * 55)


# ------------------------------------------------------------------
# 트레이너
# ------------------------------------------------------------------

class KoBARTTrainer:
    """KoBART (Seq2Seq) 학습 트레이너.

    사용 예시:
        hp = Seq2SeqHyperParams.from_yaml("configs/kobart_config.yaml")
        hp.experiment.run_name = "kobart-lr3e5-beam4"

        trainer = KoBARTTrainer(hp, notion_logger=notion, wandb_logger=wandb)
        scores = trainer.run(train_df, dev_df)
    """

    def __init__(
        self,
        hp: Seq2SeqHyperParams,
        notion_logger=None,
        wandb_logger=None,
    ):
        self.hp = hp
        self.notion = notion_logger
        self.wandb = wandb_logger
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    # 메인 실행
    # ------------------------------------------------------------------

    def run(self, train_df: pd.DataFrame, dev_df: pd.DataFrame) -> dict:
        self.hp.summary()

        if self.notion:
            self._notion_create()

        try:
            self._load_model()
            train_ds, dev_ds = self._build_datasets(train_df, dev_df)
            self._train(train_ds, dev_ds)
            scores = self._evaluate(dev_df)

            if self.wandb:
                self.wandb.log({
                    "eval/rouge1": scores["rouge1"],
                    "eval/rouge2": scores["rouge2"],
                    "eval/rougeL": scores["rougeL"],
                    "eval/score":  scores["score"],
                })
                self.wandb.log_summary("best_score", scores["score"])

            if self.notion:
                self._notion_update(scores)

            return scores

        except Exception as e:
            if self.notion:
                self.notion.fail_run(error_msg=str(e))
            raise

    # ------------------------------------------------------------------
    # 단계별 메서드
    # ------------------------------------------------------------------

    def _load_model(self):
        cfg = self.hp.model
        print(f"\n[1/3] 모델 로드: {cfg.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"      총 파라미터: {total:,}")

    def _build_datasets(
        self,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
    ) -> tuple[Dataset, Dataset]:
        print(f"\n[2/3] 데이터셋 토크나이즈: train={len(train_df):,} / dev={len(dev_df):,}")
        train_ds = self._tokenize(train_df)
        dev_ds   = self._tokenize(dev_df)
        return train_ds, dev_ds

    def _tokenize(self, df: pd.DataFrame) -> Dataset:
        cfg = self.hp.model

        def _encode(batch):
            model_inputs = self.tokenizer(
                batch["dialogue"],
                max_length=cfg.max_input_length,
                truncation=True,
                padding=False,
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    batch["summary"],
                    max_length=cfg.max_output_length,
                    truncation=True,
                    padding=False,
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = Dataset.from_pandas(df[["dialogue", "summary"]].reset_index(drop=True))
        return dataset.map(_encode, batched=True, remove_columns=["dialogue", "summary"])

    def _train(self, train_ds: Dataset, dev_ds: Dataset):
        t = self.hp.train
        exp = self.hp.experiment
        gen = self.hp.generation
        output_dir = str(Path(exp.output_dir) / exp.experiment_id)

        print(f"\n[3/3] 학습 시작")

        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=t.num_epochs,
            per_device_train_batch_size=t.batch_size,
            per_device_eval_batch_size=t.eval_batch_size,
            gradient_accumulation_steps=t.grad_accum,
            learning_rate=t.learning_rate,
            warmup_ratio=t.warmup_ratio,
            weight_decay=t.weight_decay,
            fp16=t.fp16,
            label_smoothing_factor=t.label_smoothing,
            seed=t.seed,
            # 평가 / 저장
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            # 생성 설정
            predict_with_generate=True,
            generation_num_beams=gen.num_beams,
            generation_max_length=gen.max_new_tokens,
            # 로깅
            logging_steps=50,
            report_to="wandb" if self.wandb else "none",
            run_name=exp.run_name,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,
        )

        callbacks = []
        if t.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=t.early_stopping_patience))
            print(f"      Early Stopping patience={t.early_stopping_patience}")

        self.seq2seq_trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks if callbacks else None,
        )

        self.seq2seq_trainer.train()

    def _compute_metrics(self, eval_preds) -> dict:
        """학습 중 실시간 ROUGE 계산 (토크나이저 기반, 빠른 버전)."""
        from rouge import Rouge
        rouge_scorer = Rouge()

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds  = np.where(preds  != -100, preds,  self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds  = self.tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds  = [p.strip() or "없음" for p in decoded_preds]
        decoded_labels = [l.strip() or "없음" for l in decoded_labels]

        try:
            scores = rouge_scorer.get_scores(decoded_preds, decoded_labels, avg=True)
            return {
                "rouge1": round(scores["rouge-1"]["f"], 4),
                "rouge2": round(scores["rouge-2"]["f"], 4),
                "rougeL": round(scores["rouge-l"]["f"], 4),
            }
        except Exception:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def _evaluate(self, dev_df: pd.DataFrame) -> dict:
        """MeCab 기반 공식 ROUGE 평가."""
        from src.utils.rouge_evaluator import RougeEvaluator

        print("\n[평가] MeCab ROUGE 계산 중...")
        gen = self.hp.generation
        predictions = []

        self.model.eval()
        for dialogue in dev_df["dialogue"]:
            inputs = self.tokenizer(
                dialogue,
                max_length=self.hp.model.max_input_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    num_beams=gen.num_beams,
                    length_penalty=gen.length_penalty,
                    no_repeat_ngram_size=gen.no_repeat_ngram_size,
                    max_new_tokens=gen.max_new_tokens,
                )
            pred = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            predictions.append(pred)

        evaluator = RougeEvaluator()
        scores = evaluator.score(predictions, dev_df["summary"].tolist())
        print(f"  ROUGE-1: {scores['rouge1']:.4f}")
        print(f"  ROUGE-2: {scores['rouge2']:.4f}")
        print(f"  ROUGE-L: {scores['rougeL']:.4f}")
        print(f"  Score  : {scores['score']:.4f}")
        return scores

    def predict(self, dialogues: list[str]) -> list[str]:
        """test 데이터 추론."""
        gen = self.hp.generation
        predictions = []
        self.model.eval()

        for dialogue in dialogues:
            inputs = self.tokenizer(
                dialogue,
                max_length=self.hp.model.max_input_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    num_beams=gen.num_beams,
                    length_penalty=gen.length_penalty,
                    no_repeat_ngram_size=gen.no_repeat_ngram_size,
                    max_new_tokens=gen.max_new_tokens,
                )
            pred = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            predictions.append(pred)

        return predictions

    # ------------------------------------------------------------------
    # Notion 헬퍼
    # ------------------------------------------------------------------

    def _notion_create(self):
        exp = self.hp.experiment
        t = self.hp.train
        self.notion.create_run(
            실험명=exp.run_name,
            run_name=exp.run_name,
            experiment_id=exp.experiment_id,
            dataset=exp.dataset,
            task="Summarization",
            목적=exp.purpose or (
                f"model={self.hp.model.model_name} | "
                f"lr={t.learning_rate} | epochs={t.num_epochs} | "
                f"beams={self.hp.generation.num_beams}"
            ),
            태그=exp.tags,
        )

    def _notion_update(self, scores: dict):
        summary = (
            f"ROUGE-1={scores['rouge1']:.4f} | "
            f"ROUGE-2={scores['rouge2']:.4f} | "
            f"ROUGE-L={scores['rougeL']:.4f} | "
            f"Score={scores['score']:.4f}"
        )
        self.notion.update_run(
            rouge1=scores["rouge1"],
            rouge2=scores["rouge2"],
            rougeL=scores["rougeL"],
            결과_요약=summary,
            상태="완료",
        )


# ------------------------------------------------------------------
# MT5Trainer
# ------------------------------------------------------------------

class MT5Trainer(KoBARTTrainer):
    """google/mt5-base 등 mT5 계열 Seq2Seq 학습 트레이너.

    KoBARTTrainer를 상속하며 _tokenize만 오버라이드하여
    deprecated as_target_tokenizer 대신 text_target API를 사용합니다.

    사용 예시:
        hp = Seq2SeqHyperParams.from_yaml("configs/mt5_config.yaml")
        hp.experiment.run_name = "mt5-lr5e4-ep10"

        trainer = MT5Trainer(hp, notion_logger=notion, wandb_logger=wandb)
        scores = trainer.run(train_df, dev_df)
    """

    def _tokenize(self, df: pd.DataFrame) -> Dataset:
        cfg = self.hp.model

        def _encode(batch):
            model_inputs = self.tokenizer(
                batch["dialogue"],
                max_length=cfg.max_input_length,
                truncation=True,
                padding=False,
            )
            labels = self.tokenizer(
                text_target=batch["summary"],
                max_length=cfg.max_output_length,
                truncation=True,
                padding=False,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = Dataset.from_pandas(df[["dialogue", "summary"]].reset_index(drop=True))
        return dataset.map(_encode, batched=True, remove_columns=["dialogue", "summary"])
