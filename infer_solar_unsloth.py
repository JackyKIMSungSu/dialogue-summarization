"""
SOLAR-10.7B LoRA 체크포인트로 dev 평가 + test 제출파일 생성
unsloth FastLanguageModel 사용으로 빠른 추론
"""
import sys
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

# ── 인자 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="outputs/checkpoints/EXP-20260326-050841/checkpoint-778")
parser.add_argument("--base-model", default="upstage/SOLAR-10.7B-Instruct-v1.0")
parser.add_argument("--dev-path",  default="data/raw/dev.csv")
parser.add_argument("--test-path", default="data/raw/test.csv")
parser.add_argument("--run-name",  default="solar-r16-lr2e4-ep1-newprompt")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--eval-samples", type=int, default=499, help="dev 평가 샘플 수 (0=전체)")
parser.add_argument("--no-eval", action="store_true")
parser.add_argument("--no-submission", action="store_true")
args = parser.parse_args()

PROMPT = """### System:
당신은 한국어 대화 요약 전문가입니다. 대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다. 요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요. 핵심 내용만 1~3문장으로 간결하게 요약하세요. 반드시 한국어로 작성하되, CPU·USB 등 고유 영문 약어나 전문 용어는 그대로 사용해도 됩니다.

### User:
아래 대화를 읽고 핵심 내용을 한국어로 요약해주세요. 화자 태그(#Person1# 등)를 유지하세요.

{dialogue}

### Assistant:
"""

# ── 모델 로드 ─────────────────────────────────────────────────────────
print(f"\n[1/3] 모델 로드: {args.base_model}", flush=True)
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base_model,
    max_seq_length=2048,
    load_in_4bit=True,
)

print(f"[1/3] LoRA 어댑터 로드: {args.checkpoint}", flush=True)
model = PeftModel.from_pretrained(model, args.checkpoint)
FastLanguageModel.for_inference(model)
model.eval()
print("[1/3] 로드 완료\n", flush=True)


def generate(dialogue: str) -> str:
    prompt = PROMPT.format(dialogue=dialogue)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


# ── Dev 평가 ──────────────────────────────────────────────────────────
if not args.no_eval:
    from src.utils.rouge_evaluator import RougeEvaluator

    dev_df = pd.read_csv(args.dev_path)
    n = len(dev_df) if args.eval_samples <= 0 else min(args.eval_samples, len(dev_df))
    sample_df = dev_df.sample(n=n, random_state=42)
    print(f"[2/3] Dev 평가 ({n}/{len(dev_df)}개)", flush=True)

    preds, refs = [], []
    for _, row in tqdm(sample_df.iterrows(), total=n, desc="eval"):
        preds.append(generate(row["dialogue"]))
        refs.append(row["summary"])

    scores = RougeEvaluator().score(preds, refs)
    print(f"\n  ROUGE-1: {scores['rouge1']:.4f}", flush=True)
    print(f"  ROUGE-2: {scores['rouge2']:.4f}", flush=True)
    print(f"  ROUGE-L: {scores['rougeL']:.4f}", flush=True)
    print(f"  Score  : {scores['score']:.4f}\n", flush=True)

# ── Test 제출 ─────────────────────────────────────────────────────────
if not args.no_submission:
    test_df = pd.read_csv(args.test_path)
    print(f"[3/3] Test 추론 ({len(test_df)}개)", flush=True)

    predictions = []
    for dialogue in tqdm(test_df["dialogue"], desc="test"):
        predictions.append(generate(dialogue))

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_name}.csv"

    result_df = pd.DataFrame({"fname": test_df["fname"], "summary": predictions})
    result_df.to_csv(out_path, index=False)
    print(f"[3/3] 저장 완료 → {out_path}  ({len(result_df)}행)", flush=True)
