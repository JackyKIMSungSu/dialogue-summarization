"""
제출 파일에서 영어로 된 요약을 찾아 한국어 강제 프롬프트로 재추론 후 교체
"""
import re
import sys
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

PRED_PATH   = "outputs/predictions/solar-r16-lr2e4-ep1-newprompt.csv"
TEST_PATH   = "data/raw/test.csv"
CHECKPOINT  = "outputs/checkpoints/EXP-20260326-050841/checkpoint-778"
BASE_MODEL  = "upstage/SOLAR-10.7B-Instruct-v1.0"
MAX_NEW_TOKENS = 256

# 한국어 강제 프롬프트 (기존보다 훨씬 강하게)
PROMPT_KO = """### System:
당신은 한국어 대화 요약 전문가입니다. 대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다. 요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요. 핵심 내용만 1~3문장으로 간결하게 요약하세요. 반드시 한국어로 작성하되, CPU·USB 등 고유 영문 약어나 전문 용어는 그대로 사용해도 됩니다.

### User:
아래 대화를 읽고 핵심 내용을 한국어로 요약해주세요. 화자 태그(#Person1# 등)를 유지하세요.

{dialogue}

### Assistant:
"""


def is_korean(text: str) -> bool:
    """한글 비율 30% 이상이면 정상 한국어로 판단.
    순수 영어(0%)뿐 아니라 '은/를' 등 조사만 붙은 러시아어·스페인어 혼합도 포착."""
    t = str(text)
    total = len(re.sub(r"[\s#\d\.\,\!\?\:\;\(\)\-\']", "", t))
    if total == 0:
        return True
    return len(re.findall(r"[가-힣]", t)) / total >= 0.3


# ── 모델 로드 ────────────────────────────────────────────────────────
print("[1/4] 모델 로드 중...", flush=True)
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL, max_seq_length=2048, load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, CHECKPOINT)
FastLanguageModel.for_inference(model)
model.eval()
print("[1/4] 완료\n", flush=True)


def generate(dialogue: str) -> str:
    prompt = PROMPT_KO.format(dialogue=dialogue)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


# ── 파일 로드 ────────────────────────────────────────────────────────
print("[2/4] 데이터 로드...", flush=True)
test_df = pd.read_csv(TEST_PATH)
pred_df = pd.read_csv(PRED_PATH)
merged  = test_df.merge(pred_df, on="fname")

eng_mask = ~merged["summary"].apply(is_korean)
eng_df   = merged[eng_mask].copy()
print(f"[2/4] 영어 출력 {len(eng_df)}개 / 전체 {len(merged)}개\n", flush=True)

# ── 재추론 ───────────────────────────────────────────────────────────
print(f"[3/4] 재추론 시작 ({len(eng_df)}개)", flush=True)
new_preds = {}
still_english = []

for _, row in tqdm(eng_df.iterrows(), total=len(eng_df), desc="retry"):
    pred = generate(row["dialogue"])
    new_preds[row["fname"]] = pred
    if not is_korean(pred):
        still_english.append(row["fname"])

print(f"\n  재추론 후에도 영어: {len(still_english)}개", flush=True)
if still_english:
    print(f"  → {still_english[:10]}", flush=True)

# ── 제출 파일 교체 ───────────────────────────────────────────────────
print("\n[4/4] 제출 파일 업데이트...", flush=True)
pred_df["summary"] = pred_df.apply(
    lambda r: new_preds.get(r["fname"], r["summary"]), axis=1
)

out_path = Path(PRED_PATH)
backup   = out_path.with_name(out_path.stem + "_before_fix.csv")
import shutil
shutil.copy(out_path, backup)
print(f"  백업 저장: {backup}", flush=True)

pred_df.to_csv(out_path, index=False)

# 결과 요약
final_eng = pred_df[~pred_df["summary"].apply(is_korean)]
print(f"  수정 전 영어: {len(eng_df)}개 → 수정 후 영어: {len(final_eng)}개", flush=True)
print(f"[4/4] 저장 완료: {out_path}", flush=True)
