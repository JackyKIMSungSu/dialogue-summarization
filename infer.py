"""
SOLAR 체크포인트 추론 스크립트 (배치 추론).
PEFT 어댑터를 로드하여 dev 평가 + test 제출 파일 생성.

사용 예시:
    python infer.py --checkpoint outputs/checkpoints/EXP-20260324-111751/checkpoint-1557 \
                    --run-name solar-r16-lr2e4-ep2
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


SOLAR_PROMPT = """### User:
당신은 한국어 대화 요약 전문가입니다. 대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다. 요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요. 핵심 내용만 1~3문장으로 간결하게 요약하세요.

아래 대화를 읽고 핵심 내용을 한국어로 요약해주세요. 화자 태그(#Person1# 등)를 유지하세요.

{dialogue}

### Assistant:
"""


def parse_args():
    parser = argparse.ArgumentParser(description="SOLAR 체크포인트 추론 (배치)")
    parser.add_argument("--checkpoint", required=True, help="PEFT 어댑터 체크포인트 경로")
    parser.add_argument("--run-name", default=None, help="제출 파일명 (확장자 제외)")
    parser.add_argument("--config", default="configs/solar_config.yaml", help="YAML 설정 파일")
    parser.add_argument("--dev-path", default="data/raw/dev.csv")
    parser.add_argument("--test-path", default="data/raw/test.csv")
    parser.add_argument("--batch-size", type=int, default=4, help="추론 배치 크기 (기본 4)")
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
    # decoder-only 모델은 왼쪽 패딩 필요
    tokenizer.padding_side = "left"

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


def generate_batch(model, tokenizer, dialogues: list, max_new_tokens: int, batch_size: int) -> list:
    predictions = []
    total = len(dialogues)
    t0 = time.time()

    for start in range(0, total, batch_size):
        batch = dialogues[start:start + batch_size]
        prompts = [SOLAR_PROMPT.format(dialogue=d) for d in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            pred = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            predictions.append(pred)

        done = min(start + batch_size, total)
        elapsed = time.time() - t0
        speed = done / elapsed
        eta = (total - done) / speed if speed > 0 else 0
        print(f"  {done}/{total}  ({speed:.2f} 샘플/초,  남은시간 약 {eta/60:.1f}분)", flush=True)

    return predictions


def main():
    args = parse_args()

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
        print(f"\n[Dev 평가] {len(dev_df)}개 샘플  (배치={args.batch_size})")
        dev_preds = generate_batch(model, tokenizer, dev_df["dialogue"].tolist(), max_new_tokens, args.batch_size)

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
        print(f"\n[Test 추론] {len(test_df)}개 샘플  (배치={args.batch_size})")
        test_preds = generate_batch(model, tokenizer, test_df["dialogue"].tolist(), max_new_tokens, args.batch_size)

        gen = SubmissionGenerator(
            sample_submission_path="data/raw/sample_submission.csv",
            output_dir="outputs/predictions",
        )
        path = gen.save(predictions=test_preds, filename=f"{run_name}.csv")
        print(f"제출 파일 저장 완료: {path}")


if __name__ == "__main__":
    main()
