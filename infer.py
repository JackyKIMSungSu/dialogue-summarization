"""
SOLAR 체크포인트 추론 스크립트.
PEFT 어댑터를 로드하여 dev 평가 + test 제출 파일 생성.

사용 예시:
    python infer.py --checkpoint outputs/checkpoints/EXP-20260324-111751/checkpoint-1557 \
                    --run-name solar-r16-lr2e4-ep2
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


SOLAR_PROMPT = """### User:
다음 대화를 한국어로 간결하게 요약하세요.

{dialogue}

### Assistant:
"""


def parse_args():
    parser = argparse.ArgumentParser(description="SOLAR 체크포인트 추론")
    parser.add_argument("--checkpoint", required=True, help="PEFT 어댑터 체크포인트 경로")
    parser.add_argument("--run-name", default=None, help="제출 파일명 (확장자 제외)")
    parser.add_argument("--config", default="configs/solar_config.yaml", help="YAML 설정 파일")
    parser.add_argument("--dev-path", default="data/raw/dev.csv")
    parser.add_argument("--test-path", default="data/raw/test.csv")
    parser.add_argument("--no-eval", action="store_true", help="dev 평가 건너뜀")
    parser.add_argument("--no-submission", action="store_true", help="제출 파일 생성 건너뜀")
    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    base_model_name = config["model"]["name"]
    print(f"[1/2] 베이스 모델 로드: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[2/2] PEFT 어댑터 로드: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, dialogues: list, max_new_tokens: int = 256) -> list:
    predictions = []
    total = len(dialogues)
    for i, dialogue in enumerate(dialogues):
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == total:
            print(f"  추론 중... {i+1}/{total}", flush=True)
        prompt = SOLAR_PROMPT.format(dialogue=dialogue)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        predictions.append(pred)
    return predictions


def main():
    args = parse_args()

    # dialogue-summarization 디렉토리 기준으로 실행
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    max_new_tokens = config["training"].get("max_new_tokens", 256)
    run_name = args.run_name or Path(args.checkpoint).name

    model, tokenizer = load_model(args.checkpoint, config)

    # Dev 평가
    if not args.no_eval:
        from src.utils.rouge_evaluator import RougeEvaluator

        dev_df = pd.read_csv(args.dev_path)
        print(f"\n[Dev 평가] {len(dev_df)}개 샘플")
        dev_preds = generate(model, tokenizer, dev_df["dialogue"].tolist(), max_new_tokens)

        evaluator = RougeEvaluator()
        scores = evaluator.score(dev_preds, dev_df["summary"].tolist())
        print(f"\n=== Dev ROUGE 점수 ({run_name}) ===")
        print(f"  ROUGE-1: {scores['rouge1']:.4f}")
        print(f"  ROUGE-2: {scores['rouge2']:.4f}")
        print(f"  ROUGE-L: {scores['rougeL']:.4f}")
        print(f"  Score  : {scores['score']:.4f}  (rouge1+rouge2+rougeL)")

    # Test 제출 파일 생성
    if not args.no_submission:
        from src.inference.submit import SubmissionGenerator

        test_df = pd.read_csv(args.test_path)
        print(f"\n[Test 추론] {len(test_df)}개 샘플")
        test_preds = generate(model, tokenizer, test_df["dialogue"].tolist(), max_new_tokens)

        gen = SubmissionGenerator(
            sample_submission_path="data/raw/sample_submission.csv",
            output_dir="outputs/predictions",
        )
        path = gen.save(predictions=test_preds, filename=f"{run_name}.csv")
        print(f"제출 파일 저장 완료: {path}")


if __name__ == "__main__":
    main()
