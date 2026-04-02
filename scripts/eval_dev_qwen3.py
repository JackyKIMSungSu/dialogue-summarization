"""
Qwen3-8B LoRA 체크포인트로 dev set 추론 → ROUGE 계산 → Notion 업데이트
"""
import gc, os, re, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('/data/ephemeral/home/project/nlp-team2/dialogue-summarization/.env')
sys.path.insert(0, '/data/ephemeral/home/project/nlp-team2/dialogue-summarization')

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge import Rouge
from notion_client import Client

# ── CONFIG ────────────────────────────────────────────────
MODEL_NAME   = 'Qwen/Qwen3-8B'
ADAPTER_PATH = Path('/root/upstage-nlp-nlp/code/prediction/qwen3_response_only_best_strategy_8b/qwen3-8b-backtrans-r32-lr0.0001-ep3-20260401-081433/final')
DEV_CSV      = Path('/data/ephemeral/home/project/nlp-team2/dialogue-summarization/data/raw/dev.csv')
OUT_CSV      = Path('/root/upstage-nlp-nlp/code/prediction/qwen3_response_only_best_strategy_8b/dev_predictions.csv')
MAX_SEQ_LEN  = 1536
MAX_NEW_TOKENS = 90

NOTION_PAGE_IDS = {
    '학습':       '3365930a-a61a-81f5-bea2-d92e1dc740e5',
    'submit_11': '3365930a-a61a-8179-a4cf-fb4369185e7c',
    'submit_12': '3365930a-a61a-81fb-8d14-d7796772c636',
    'submit_13': '3365930a-a61a-8189-9498-fd3e80e80ae5',
}

SYSTEM_PROMPT = (
    '당신은 한국어 대화 요약 전문가입니다. '
    '대화의 핵심 사건만 아주 짧게 요약하세요. '
    '반드시 #Person1#, #Person2# 같은 화자 태그를 유지하세요. '
    '1문장만 출력하고, 불필요한 수식어는 쓰지 마세요.'
)
USER_TMPL = (
    '다음 대화를 1문장으로만 매우 간결하게 요약하세요. '
    '핵심 행동/결정만 남기고 화자 태그는 유지하세요.\n\n{dialogue}'
)

# ── 유틸 ──────────────────────────────────────────────────
def normalize(text):
    text = str(text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'#\s*Person\s*(\d+)\s*#', r'#Person\1#', text)
    text = re.sub(r'^요약\s*:\s*', '', text).strip()
    return re.sub(r'\s+', ' ', text).strip() or '빈 요약'

def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=pick_dtype(),
    )

# ── 1. 모델 로드 ──────────────────────────────────────────
print('=== 모델 로드 ===')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=get_bnb_config(),
    device_map={'': 0},
    dtype=pick_dtype(),
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH), is_trainable=False)
model.eval()
model.config.use_cache = True
print('모델 로드 완료')

# ── 2. Dev 추론 (캐시 있으면 스킵) ───────────────────────
dev_df = pd.read_csv(DEV_CSV)
print(f'\n=== Dev 추론 ({len(dev_df)}샘플) ===')

if OUT_CSV.exists():
    cached = pd.read_csv(OUT_CSV)
    if len(cached) == len(dev_df):
        print('캐시 사용:', OUT_CSV)
        preds = [normalize(s) for s in cached['summary']]
    else:
        OUT_CSV.unlink()
        preds = None
else:
    preds = None

if preds is None:
    preds = []
    for i, row in enumerate(dev_df.itertuples(index=False), start=1):
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': USER_TMPL.format(dialogue=row.dialogue)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                repetition_penalty=1.05, use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        preds.append(normalize(pred))
        if i % 50 == 0:
            print(f'  {i}/{len(dev_df)} 완료')

    # 저장
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'fname': dev_df['fname'], 'summary': preds}).to_csv(OUT_CSV, index=False)
    print(f'Dev 예측 저장: {OUT_CSV}')

# ── 3. ROUGE 계산 ─────────────────────────────────────────
print('\n=== ROUGE 계산 ===')
rouge_scorer = Rouge()
refs = [normalize(s) for s in dev_df['summary']]

scores = rouge_scorer.get_scores(preds, refs, avg=True)
rouge1 = scores['rouge-1']['f']
rouge2 = scores['rouge-2']['f']
rougeL = scores['rouge-l']['f']
final  = rouge1 + rouge2 + rougeL

print(f'  rouge-1 : {rouge1:.4f}')
print(f'  rouge-2 : {rouge2:.4f}')
print(f'  rouge-L : {rougeL:.4f}')
print(f'  합계    : {final:.4f}')

# ── 4. Notion 업데이트 ────────────────────────────────────
print('\n=== Notion 업데이트 ===')
notion = Client(auth=os.environ['NOTION_TOKEN'])

score_summary = (
    f'[Dev ROUGE] rouge1={rouge1:.4f}, rouge2={rouge2:.4f}, '
    f'rougeL={rougeL:.4f}, 합계={final:.4f}\n'
    f'프롬프트: abstract_short_base (1문장 간결 요약)\n'
    f'데이터: train.csv 12,457 + train_backtrans.csv 12,457 = 24,914샘플\n'
    f'체크포인트: {ADAPTER_PATH}'
)

# 학습 페이지
notion.pages.update(
    page_id=NOTION_PAGE_IDS['학습'],
    properties={
        'rouge1':       {'number': round(rouge1, 4)},
        'rouge2':       {'number': round(rouge2, 4)},
        'rougeL':       {'number': round(rougeL, 4)},
        'final_result': {'number': round(final, 4)},
        '결과 요약':    {'rich_text': [{'text': {'content': score_summary[:2000]}}]},
    }
)
print('[학습] Notion 업데이트 완료')

# submit_11/12/13 - 동일 dev 점수 기록 (test 점수는 리더보드 제출 후)
for key, pid in [('submit_11', NOTION_PAGE_IDS['submit_11']),
                 ('submit_12', NOTION_PAGE_IDS['submit_12']),
                 ('submit_13', NOTION_PAGE_IDS['submit_13'])]:
    notion.pages.update(
        page_id=pid,
        properties={
            'rouge1':       {'number': round(rouge1, 4)},
            'rouge2':       {'number': round(rouge2, 4)},
            'rougeL':       {'number': round(rougeL, 4)},
            'final_result': {'number': round(final, 4)},
        }
    )
    print(f'[{key}] Notion 업데이트 완료')

# GPU 해제
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

print('\n===== 완료 =====')
print(f'Dev ROUGE — rouge1: {rouge1:.4f}, rouge2: {rouge2:.4f}, rougeL: {rougeL:.4f}, 합계: {final:.4f}')
