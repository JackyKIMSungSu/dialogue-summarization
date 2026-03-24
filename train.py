"""
대화 요약 모델 학습 CLI

사용 예시:
    python train.py --model kobart
    python train.py --model mt5 --run-name mt5-exp01 --epochs 5
    python train.py --model solar --run-name solar-r16 --lr 2e-4
    python train.py --model kobart --config configs/kobart_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv(".env")


# ------------------------------------------------------------------
# 인수 파싱
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="대화 요약 모델 학습 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모델 종류:
  kobart   gogamza/kobart-base-v2         Seq2Seq, ~4GB VRAM
  mt5      google/mt5-base                Seq2Seq, ~4GB VRAM
  solar    upstage/SOLAR-10.7B-Instruct   Causal LM + QLoRA, ~14GB VRAM

예시:
  python train.py --model kobart --run-name kobart-exp01
  python train.py --model mt5 --epochs 5 --lr 5e-4
  python train.py --model solar --run-name solar-r16 --lora-r 16
        """,
    )

    parser.add_argument(
        "--model", required=True, choices=["kobart", "mt5", "solar"],
        help="학습할 모델 종류",
    )
    parser.add_argument(
        "--config", default=None,
        help="YAML 설정 파일 경로 (미지정 시 기본 config 사용)",
    )
    parser.add_argument("--run-name", default=None, help="실험 이름 (WandB / Notion)")
    parser.add_argument("--purpose", default="", help="실험 목적 설명")
    parser.add_argument("--tags", nargs="*", default=None, help="태그 목록")

    # 공통 학습 오버라이드
    parser.add_argument("--epochs", type=int, default=None, help="학습 에폭 수")
    parser.add_argument("--lr", type=float, default=None, help="학습률")
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")

    # Seq2Seq 전용
    parser.add_argument("--num-beams", type=int, default=None, help="빔 서치 크기 (kobart/mt5)")

    # LoRA 전용 (solar)
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank (solar)")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (solar)")
    parser.add_argument("--grad-accum", type=int, default=None, help="gradient accumulation steps (solar)")

    # 데이터 경로
    parser.add_argument("--train-path", default="data/raw/train.csv", help="학습 데이터 경로")
    parser.add_argument("--dev-path", default="data/raw/dev.csv", help="검증 데이터 경로")
    parser.add_argument("--test-path", default="data/raw/test.csv", help="테스트 데이터 경로")

    # 출력
    parser.add_argument("--output-dir", default="outputs/checkpoints", help="체크포인트 저장 경로")
    parser.add_argument("--no-submission", action="store_true", help="제출 파일 생성 건너뜀")
    parser.add_argument("--no-wandb", action="store_true", help="WandB 로깅 비활성화")
    parser.add_argument("--no-notion", action="store_true", help="Notion 로깅 비활성화")

    return parser.parse_args()


# ------------------------------------------------------------------
# WandB / Notion 초기화
# ------------------------------------------------------------------

def init_loggers(args, config: dict, run_name: str, tags: list):
    from src.utils.wandb_logger import WandbLogger
    from src.utils.notion_logger import NotionLogger

    wandb_logger = None
    notion_logger = None

    if not args.no_wandb:
        config["wandb"]["name"] = run_name
        config["wandb"]["tags"] = tags
        wandb_logger = WandbLogger(config)
        print(f"WandB: {wandb_logger.run.url}")

    if not args.no_notion:
        notion_logger = NotionLogger()

    return wandb_logger, notion_logger


# ------------------------------------------------------------------
# 제출 파일 생성
# ------------------------------------------------------------------

def save_submission(predictions: list[str], run_name: str):
    from src.inference.submit import SubmissionGenerator
    gen = SubmissionGenerator(
        sample_submission_path="data/raw/sample_submission.csv",
        output_dir="outputs/predictions",
    )
    path = gen.save(predictions=predictions, filename=f"{run_name}.csv")
    print(f"제출 파일: {path}")


# ------------------------------------------------------------------
# KoBART / mt5 학습
# ------------------------------------------------------------------

def run_seq2seq(args, model_key: str):
    from src.training.seq2seq_trainer import (
        Seq2SeqModelConfig, Seq2SeqTrainConfig,
        GenerationConfig, Seq2SeqExperimentConfig,
        Seq2SeqHyperParams, KoBARTTrainer, MT5Trainer,
    )

    config_path = args.config or f"configs/{model_key}_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 기본 HyperParams
    hp = Seq2SeqHyperParams.from_yaml(config_path)

    # 오버라이드
    if args.epochs is not None:
        hp.train.num_epochs = args.epochs
    if args.lr is not None:
        hp.train.learning_rate = args.lr
    if args.batch_size is not None:
        hp.train.batch_size = args.batch_size
    if args.seed is not None:
        hp.train.seed = args.seed
    if args.num_beams is not None:
        hp.generation.num_beams = args.num_beams

    default_tags = {"kobart": ["KoBART", "seq2seq"], "mt5": ["mt5", "seq2seq", "multilingual"]}
    run_name = args.run_name or f"{model_key}-lr{hp.train.learning_rate}-ep{hp.train.num_epochs}"
    tags = args.tags or default_tags.get(model_key, [])

    hp.experiment.run_name = run_name
    hp.experiment.purpose = args.purpose
    hp.experiment.tags = tags
    hp.experiment.output_dir = args.output_dir

    wandb_logger, notion_logger = init_loggers(args, config, run_name, tags)

    # 데이터 로드
    train_df = pd.read_csv(args.train_path)
    dev_df   = pd.read_csv(args.dev_path)
    test_df  = pd.read_csv(args.test_path)
    print(f"train: {len(train_df):,}  |  dev: {len(dev_df):,}  |  test: {len(test_df):,}")

    TrainerClass = MT5Trainer if model_key == "mt5" else KoBARTTrainer
    trainer = TrainerClass(hp=hp, notion_logger=notion_logger, wandb_logger=wandb_logger)
    scores = trainer.run(train_df, dev_df)

    print("\n=== 최종 ROUGE 점수 (MeCab) ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")

    if not args.no_submission:
        predictions = trainer.predict(test_df["dialogue"].tolist())
        save_submission(predictions, run_name)

    if wandb_logger:
        wandb_logger.finish()


# ------------------------------------------------------------------
# SOLAR 학습
# ------------------------------------------------------------------

def run_solar(args):
    from src.training.trainer import (
        ModelConfig, LoraConfig, TrainConfig,
        ExperimentConfig, HyperParams,
        DialogueSummarizationTrainer,
    )

    config_path = args.config or "configs/solar_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    lc = config["lora"]
    tc = config["training"]

    model_cfg = ModelConfig(
        model_name=mc["name"],
        max_seq_length=mc["max_seq_length"],
        load_in_4bit=mc["load_in_4bit"],
    )
    lora_cfg = LoraConfig(
        r=lc["r"],
        alpha=lc["alpha"],
        dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        bias=lc["bias"],
    )
    train_cfg = TrainConfig(
        num_epochs=tc["num_epochs"],
        batch_size=tc["batch_size"],
        grad_accum=tc["grad_accum"],
        learning_rate=tc["learning_rate"],
        lr_scheduler=tc["lr_scheduler"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        bf16=tc["bf16"],
        seed=tc["seed"],
        response_only=tc["response_only"],
        max_new_tokens=tc["max_new_tokens"],
        instruction_part=tc["instruction_part"],
        response_part=tc["response_part"],
    )

    # 오버라이드
    if args.epochs is not None:
        train_cfg.num_epochs = args.epochs
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.lora_r is not None:
        lora_cfg.r = args.lora_r
    if args.lora_alpha is not None:
        lora_cfg.alpha = args.lora_alpha
    if args.grad_accum is not None:
        train_cfg.grad_accum = args.grad_accum

    tags = args.tags or ["SOLAR", "causal-lm", "QLoRA"]
    run_name = args.run_name or f"solar-r{lora_cfg.r}-lr{train_cfg.learning_rate}-ep{train_cfg.num_epochs}"

    exp_cfg = ExperimentConfig(
        run_name=run_name,
        dataset="DialogSum-KO",
        purpose=args.purpose,
        tags=tags,
        output_dir=args.output_dir,
    )

    hp = HyperParams(model=model_cfg, lora=lora_cfg, train=train_cfg, experiment=exp_cfg)
    hp.summary()

    wandb_logger, notion_logger = init_loggers(args, config, run_name, tags)

    # 데이터 로드
    train_df = pd.read_csv(args.train_path)
    dev_df   = pd.read_csv(args.dev_path)
    test_df  = pd.read_csv(args.test_path)
    print(f"train: {len(train_df):,}  |  dev: {len(dev_df):,}  |  test: {len(test_df):,}")

    SOLAR_PROMPT = """### User:
다음 대화를 한국어로 간결하게 요약하세요.

{dialogue}

### Assistant:
{summary}"""

    trainer = DialogueSummarizationTrainer(
        hp=hp,
        prompt_template=SOLAR_PROMPT,
        notion_logger=notion_logger,
        wandb_logger=wandb_logger,
    )
    scores = trainer.run(train_df, dev_df)

    print("\n=== 최종 ROUGE 점수 (MeCab) ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")

    if not args.no_submission:
        import torch
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(trainer.model)
        predictions = []
        for dialogue in test_df["dialogue"]:
            prompt = SOLAR_PROMPT.format(dialogue=dialogue, summary="")
            inputs = trainer.tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = trainer.model.generate(
                    **inputs,
                    max_new_tokens=hp.train.max_new_tokens,
                    do_sample=False,
                )
            pred = trainer.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            predictions.append(pred)
        save_submission(predictions, run_name)

    if wandb_logger:
        wandb_logger.finish()


# ------------------------------------------------------------------
# 진입점
# ------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'='*55}")
    print(f"  모델: {args.model.upper()}")
    if args.run_name:
        print(f"  실험: {args.run_name}")
    print(f"{'='*55}\n")

    if args.model in ("kobart", "mt5"):
        run_seq2seq(args, args.model)
    elif args.model == "solar":
        run_solar(args)
