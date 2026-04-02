"""
대화 요약 학습 트레이너.

Unsloth + QLoRA + SFTTrainer 기반.
WandB / Notion 연동 포함.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
import yaml


# ------------------------------------------------------------------
# 하이퍼파라미터 설정
# ------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_name: str = "unsloth/Qwen3-14B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True          # QLoRA 4-bit 양자화


@dataclass
class LoraConfig:
    r: int = 16                        # LoRA rank (8/16/32/64)
    alpha: int = 16                    # LoRA alpha (보통 r과 동일)
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"


@dataclass
class TrainConfig:
    num_epochs: int = 3
    batch_size: int = 1                # per_device_train_batch_size
    grad_accum: int = 32              # gradient_accumulation_steps
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"       # cosine / linear / constant
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    seed: int = 42
    response_only: bool = True         # True: 요약 부분만 loss 계산 (권장)
    max_new_tokens: int = 256          # 추론 시 최대 생성 토큰
    instruction_part: str = "[대화]\n"  # response_only 시 instruction 마커
    response_part: str = "[요약]\n"     # response_only 시 response 마커


@dataclass
class ExperimentConfig:
    experiment_id: str = ""
    run_name: str = ""
    dataset: str = "DialogSum-KO"
    purpose: str = ""
    tags: list[str] = field(default_factory=list)
    output_dir: str = "outputs/checkpoints"
    prediction_dir: str = "outputs/predictions"

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = datetime.now().strftime("EXP-%Y%m%d-%H%M%S")
        if not self.run_name:
            self.run_name = self.experiment_id


@dataclass
class HyperParams:
    """전체 실험 설정을 하나로 묶은 클래스."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> dict:
        return {
            "model":      asdict(self.model),
            "lora":       asdict(self.lora),
            "train":      asdict(self.train),
            "experiment": asdict(self.experiment),
        }

    def effective_batch_size(self) -> int:
        return self.train.batch_size * self.train.grad_accum

    def summary(self):
        print("=" * 55)
        print(f"  실험 ID   : {self.experiment.experiment_id}")
        print(f"  Run Name  : {self.experiment.run_name}")
        print(f"  모델      : {self.model.model_name}")
        print(f"  LoRA rank : {self.lora.r}  |  alpha: {self.lora.alpha}")
        print(f"  LR        : {self.train.learning_rate}  |  Epochs: {self.train.num_epochs}")
        print(f"  실효 배치 : {self.train.batch_size} × {self.train.grad_accum} = {self.effective_batch_size()}")
        print(f"  4-bit     : {self.model.load_in_4bit}  |  BF16: {self.train.bf16}")
        print(f"  응답만 학습: {self.train.response_only}")
        print("=" * 55)


# ------------------------------------------------------------------
# 트레이너
# ------------------------------------------------------------------

class DialogueSummarizationTrainer:
    """대화 요약 학습 트레이너.

    사용 예시:
        hp = HyperParams(
            model=ModelConfig(model_name="unsloth/Qwen3-14B"),
            lora=LoraConfig(r=32),
            train=TrainConfig(learning_rate=1e-4, num_epochs=5),
            experiment=ExperimentConfig(run_name="exp-lora-r32"),
        )
        trainer = DialogueSummarizationTrainer(hp)
        trainer.run(train_df, dev_df)
    """

    def __init__(
        self,
        hp: HyperParams,
        prompt_template: str | None = None,
        notion_logger=None,
        wandb_logger=None,
        resume_from_checkpoint: str | None = None,
    ):f
        self.hp = hp
        self.prompt_template = prompt_template or _DEFAULT_PROMPT
        self.notion = notion_logger
        self.wandb = wandb_logger
        self.resume_from_checkpoint = resume_from_checkpoint

        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    # 메인 실행
    # ------------------------------------------------------------------

    def run(self, train_df: pd.DataFrame, dev_df: pd.DataFrame):
        """전체 학습 파이프라인 실행."""
        hp = self.hp
        hp.summary()

        # Notion: 실험 시작 기록
        if self.notion:
            self._notion_create()

        try:
            self._load_model()
            self._apply_lora()
            train_dataset = self._build_dataset(train_df)
            self._train(train_dataset)
            rouge_scores = self._evaluate(dev_df)

            # 체크포인트 저장
            ckpt_path = self._save_checkpoint()

            # WandB: ROUGE 점수 로깅
            if self.wandb:
                self.wandb.log({
                    "eval/rouge1": rouge_scores["rouge1"],
                    "eval/rouge2": rouge_scores["rouge2"],
                    "eval/rougeL": rouge_scores["rougeL"],
                    "eval/score":  rouge_scores["score"],
                })
                self.wandb.log_summary("best_score", rouge_scores["score"])

            # Notion: 결과 업데이트
            if self.notion:
                self._notion_update(rouge_scores)

            return rouge_scores

        except Exception as e:
            if self.notion:
                self.notion.fail_run(error_msg=str(e))
            raise

    # ------------------------------------------------------------------
    # 단계별 메서드
    # ------------------------------------------------------------------

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        cfg = self.hp.model
        print(f"\n[1/4] 모델 로드: {cfg.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if cfg.load_in_4bit else None

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
        total = sum(p.numel() for p in self.model.parameters())
        print(f"      총 파라미터: {total:,}")

    def _apply_lora(self):
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
        lora = self.hp.lora
        print(f"\n[2/4] LoRA 적용 (r={lora.r}, alpha={lora.alpha})")

        self.model = prepare_model_for_kbit_training(self.model)
        self.model.enable_input_require_grads()

        peft_config = PeftLoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            target_modules=lora.target_modules,
            lora_dropout=lora.dropout,
            bias=lora.bias,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"      학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _build_dataset(self, df: pd.DataFrame):
        from datasets import Dataset
        print(f"\n[3/4] 데이터셋 구성: {len(df):,}개")

        def format_row(row):
            return {"text": self.prompt_template.format(
                dialogue=row["dialogue"],
                summary=row["summary"],
            )}

        records = [format_row(row) for _, row in df.iterrows()]
        return Dataset.from_list(records)

    def _train(self, train_dataset):
        from trl import SFTTrainer, SFTConfig

        t = self.hp.train
        exp = self.hp.experiment
        output_dir = str(Path(exp.output_dir) / exp.experiment_id)
        use_bf16 = t.bf16 and torch.cuda.is_bf16_supported()

        print(f"\n[4/4] 학습 시작 (실효 배치={self.hp.effective_batch_size()}, bf16={use_bf16})")

        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=t.num_epochs,
            per_device_train_batch_size=t.batch_size,
            gradient_accumulation_steps=t.grad_accum,
            learning_rate=t.learning_rate,
            lr_scheduler_type=t.lr_scheduler,
            warmup_ratio=t.warmup_ratio,
            weight_decay=t.weight_decay,
            max_grad_norm=t.max_grad_norm,
            bf16=use_bf16,
            fp16=not use_bf16,
            seed=t.seed,
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            report_to="wandb" if self.wandb else "none",
            run_name=exp.run_name,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            dataset_text_field="text",
            max_length=self.hp.model.max_seq_length,
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            args=training_args,
        )

        if t.response_only:
            # 요약 부분만 loss 계산: response_part 이후 토큰만 loss 적용
            response_token_ids = self.tokenizer.encode(
                t.response_part, add_special_tokens=False
            )
            instruction_token_ids = self.tokenizer.encode(
                t.instruction_part, add_special_tokens=False
            )
            trainer = _apply_response_only(
                trainer, instruction_token_ids, response_token_ids
            )

        self.trainer = trainer
        if self.resume_from_checkpoint:
            print(f"      체크포인트에서 재개: {self.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint or None)

    def _evaluate(self, dev_df: pd.DataFrame, eval_samples: int = 500) -> dict:
        import sys
        from tqdm import tqdm
        from src.utils.rouge_evaluator import RougeEvaluator

        sample_df = dev_df.sample(n=min(eval_samples, len(dev_df)), random_state=42)
        print(f"\n[평가] dev 세트 추론 중... ({len(sample_df)}/{len(dev_df)}개 샘플)", flush=True)
        if self.hp.train.bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model.eval()

        predictions = []
        refs = []
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="eval"):
            prompt = self.prompt_template.format(dialogue=row["dialogue"], summary="")
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.hp.train.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            pred = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            predictions.append(pred)
            refs.append(row["summary"])

        evaluator = RougeEvaluator()
        scores = evaluator.score(predictions, refs)
        print(f"  ROUGE-1: {scores['rouge1']:.4f}", flush=True)
        print(f"  ROUGE-2: {scores['rouge2']:.4f}", flush=True)
        print(f"  ROUGE-L: {scores['rougeL']:.4f}", flush=True)
        print(f"  Score  : {scores['score']:.4f}", flush=True)
        return scores

    def _save_checkpoint(self) -> str:
        exp = self.hp.experiment
        ckpt_path = str(Path(exp.output_dir) / exp.experiment_id / "final")
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        print(f"\n[저장] 체크포인트 → {ckpt_path}")
        return ckpt_path

    # ------------------------------------------------------------------
    # Notion 헬퍼
    # ------------------------------------------------------------------

    def _notion_create(self):
        exp = self.hp.experiment
        hp_summary = (
            f"model={self.hp.model.model_name} | "
            f"lora_r={self.hp.lora.r} | "
            f"lr={self.hp.train.learning_rate} | "
            f"epochs={self.hp.train.num_epochs} | "
            f"batch={self.hp.effective_batch_size()}"
        )
        wandb_url = self.wandb.run.url if self.wandb else ""
        self.notion.create_run(
            실험명=exp.run_name,
            run_name=exp.run_name,
            experiment_id=exp.experiment_id,
            dataset=exp.dataset,
            task="Summarization",
            목적=exp.purpose or hp_summary,
            태그=exp.tags,
        )

    def _notion_update(self, scores: dict):
        wandb_url = self.wandb.run.url if self.wandb else ""
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
# Response-only 학습 헬퍼 (unsloth 없이 구현)
# ------------------------------------------------------------------

def _apply_response_only(trainer, instruction_ids: list, response_ids: list):
    """instruction 부분의 loss를 -100으로 마스킹하는 DataCollator를 적용."""
    import torch
    from torch.utils.data import DataLoader

    original_collator = trainer.data_collator

    class ResponseOnlyCollator:
        def __init__(self, base_collator, resp_ids):
            self.base = base_collator
            self.resp_ids = resp_ids

        def __call__(self, features):
            batch = self.base(features)
            labels = batch["labels"].clone()
            for i, label_row in enumerate(labels):
                # response_part 토큰 시작 위치 이후만 loss 유지
                resp_start = None
                ids = label_row.tolist()
                for j in range(len(ids) - len(self.resp_ids) + 1):
                    if ids[j:j+len(self.resp_ids)] == self.resp_ids:
                        resp_start = j + len(self.resp_ids)
                        break
                if resp_start is not None:
                    labels[i, :resp_start] = -100
                else:
                    labels[i] = torch.full_like(label_row, -100)
            batch["labels"] = labels
            return batch

    trainer.data_collator = ResponseOnlyCollator(original_collator, response_ids)
    return trainer


# ------------------------------------------------------------------
# 기본 프롬프트 템플릿
# ------------------------------------------------------------------

_DEFAULT_PROMPT = """다음 대화를 한국어로 간결하게 요약하세요.

[대화]
{dialogue}

[요약]
{summary}"""
