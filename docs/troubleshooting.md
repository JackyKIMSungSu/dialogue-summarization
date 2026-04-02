# 트러블슈팅 & 실험 분석

---

## [2026-03-26] SOLAR-10.7B epoch 2 체크포인트 ROUGE 점수 저조

### 실험 정보

| 항목 | 값 |
|------|----|
| Run Name | solar-r16-lr2e4-ep2 |
| Experiment ID | EXP-20260324-111751 |
| 체크포인트 | checkpoint-1557 (epoch 2 / 총 2334 steps) |
| Dev ROUGE-1 | 0.1372 |
| Dev ROUGE-2 | 0.0519 |
| Dev ROUGE-L | 0.1317 |
| **Dev Score** | **0.3208** |
| 비교 기준 | KoBART best (ROUGE-1 ~0.60) |

---

### 문제 현상

- 요약 결과가 한국어가 아닌 **영어로 생성**되는 비율이 높음
- 예시:
  - `"Person1은 Person2에게 internal memo를 all staff에게 4시전까지 distribute하라고 instructs while explaining..."` → 완전 영어
  - `"#Person2#은 traffic jam에 because of and arrived late..."` → 영어 혼용
- MeCab ROUGE 특성상 영어 출력은 한국어 정답과 n-gram이 거의 겹치지 않아 점수가 급락

---

### 원인 추정

#### 1. 프롬프트에 한국어 출력 지시가 부족했음 (주요 원인)
- 기존 프롬프트: `"다음 대화를 한국어로 간결하게 요약하세요."`
- SOLAR-10.7B-Instruct는 영어 대화 입력 시 영어로 응답하려는 경향이 강함
- 화자 태그(`#Person1#` 등) 처리 방식에 대한 지시가 없어 모델이 영어 문맥을 그대로 따라감
- **대응**: 프롬프트에 화자 태그 유지 명시, 한국어 강조, 1~3문장 제한 추가 (2026-03-26 수정)

#### 2. 학습이 epoch 2에서 중단됨 (미완성 학습)
- 총 3 epoch 계획 중 epoch 2 체크포인트(step 1557)만 사용 가능
- epoch 3 학습을 시도했으나 infer.py 완료 대기 후 재개 중 세션 전환으로 중단
- epoch 3까지 완료됐다면 한국어 요약 패턴을 더 많이 학습했을 가능성 있음

#### 3. Response-only 학습 마커 문제 가능성
- `instruction_part: "### User:\n"`, `response_part: "### Assistant:\n"` 설정으로 response-only 학습 적용
- 프롬프트가 짧아 response 비율이 높고, 프롬프트 내 한국어 지시 신호가 약했을 수 있음

#### 4. 베이스 데이터의 영어 대화 비율
- DialogSum-KO 학습 데이터가 영어 원문을 포함하고 있어, 모델이 입력 언어를 그대로 출력에 반영하는 경향
- 영어 대화 → 한국어 요약으로의 cross-lingual 매핑을 충분히 학습하지 못한 상태

---

### 조치 사항

| 조치 | 내용 | 상태 |
|------|------|------|
| 프롬프트 개선 | 화자 태그 유지, 한국어 명시, 1~3문장 제한 추가 | ✅ 완료 (2026-03-26) |
| 재학습 | 새 프롬프트로 epoch 1 학습 시작 (`solar-r16-lr2e4-ep1-newprompt`) | ✅ 완료 (2026-03-26 05:08~11:29) |

---

### 개선된 프롬프트 (2026-03-26~)

```
### User:
당신은 한국어 대화 요약 전문가입니다. 대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다. 요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요. 핵심 내용만 1~3문장으로 간결하게 요약하세요.

아래 대화를 읽고 핵심 내용을 한국어로 요약해주세요. 화자 태그(#Person1# 등)를 유지하세요.

{dialogue}

### Assistant:
```

---

## [2026-03-26] epoch 1 완료 후 평가/제출 추론 트러블슈팅

### 배경

- `solar-r16-lr2e4-ep1-newprompt` 학습 완료 (05:08~11:29, 778 steps)
- 이후 자동으로 `_evaluate()` → test 제출파일 생성 예정이었으나 진행 이상 감지

---

### 문제 1: 로그가 11:29 이후 업데이트 안 됨

**원인**: Python `print()`는 stdout full buffering 모드. 파일로 리다이렉트 시 버퍼가 가득 차거나 프로그램 종료 전까지 flush 안 됨. tqdm만 stderr → 즉시 기록됨.

**확인 방법**: `/proc/<PID>/fdinfo/1` 의 `pos` 값 = 현재 파일 크기 → 아무것도 새로 쓰지 않은 상태

**해결**: `python -u` 플래그 (unbuffered) + `print(..., flush=True)` 추가

---

### 문제 2: 평가 추론 극도로 느림 (490초/샘플)

**원인**: unsloth 미설치 상태에서 4-bit QLoRA 모델 추론 = 매 step마다 NF4 역양자화 발생. GPU 100% 사용이지만 메모리 대역폭 병목으로 ~490초/샘플.

**unsloth 미설치 시 추정 소요 시간**: dev(499개) + test(499개) = ~136시간(!)

**해결**: `unsloth` 설치 + `FastLanguageModel.for_inference()` 적용 → **38초/샘플 (13배 개선)**

---

### 문제 3: unsloth 설치 시 패키지 버전 대규모 변경

**변경 내역**:

| 패키지 | 이전 | 이후 |
|--------|------|------|
| torch | 2.1.0+cu121 | 2.10.0 |
| transformers | 4.44.0 | 5.3.0 |
| trl | 0.9.6 | 0.24.0 |
| peft | 0.12.0 | 0.18.1 |
| bitsandbytes | 0.41.3 | 0.49.2 |

**주의**: 향후 train.py의 SFTTrainer 구버전 API(`dataset_text_field`, `max_seq_length` 직접 전달 방식)가 trl 0.24에서 에러로 바뀔 수 있음. 재학습 시 `SFTConfig` 방식으로 수정 필요.

---

### 문제 4: unsloth import 시 `No module named 'unsloth'` (설치 전 코드에 import 있음)

**원인**: train.py의 test 제출 루프에서 `from unsloth import FastLanguageModel` 사용하는데 환경에 unsloth 미설치.

**해결**: 제출 루프에서 unsloth 의존성 제거, 전용 `infer_solar_unsloth.py` 스크립트 분리

---

### 문제 5: unsloth 로드 시 `RuntimeError: Failed to find C compiler`

**원인**: triton 3.6.0이 CUDA 커널 컴파일에 gcc 필요. 시스템에 gcc 미설치.

**해결**: `apt-get install -y gcc`

---

### 문제 6: GPU 메모리 부족 (이전 프로세스 잔재)

**원인**: `kill PID` 했어도 자식 프로세스가 남아 각각 ~8GB GPU 메모리 점유. `kill PPID`만으로는 자식 프로세스 미종료.

**해결**: `ps aux | grep python` 으로 모든 관련 프로세스 확인 후 전체 kill

---

### 완료 상태 (2026-03-27)

- `infer_solar_unsloth.py` 완료
- 속도: **~50초/샘플** (unsloth 적용, 워밍업 후 안정)
- dev ROUGE-1: 0.1570 / ROUGE-2: 0.1169 / ROUGE-L: 0.1492 / Score: 0.4231
- 결과: `outputs/predictions/solar-r16-lr2e4-ep1-newprompt.csv` (499행)

---

## [2026-03-27] 제출 파일 다국어 출력 문제 및 후처리

### 문제 7: 제출 파일에 영어·러시아어·스페인어 등 비한국어 요약 포함

**현황**:
- 대화는 모두 한국어임에도 499개 중 **136개(27.3%)가 순수 영어** 출력
- 추가로 한글 조사(`은/를`)만 있고 나머지가 러시아어·스페인어인 케이스 존재 (예: test_19)
  - `#Person1#은 #Person2#를 помочь просит. #Person2# хочет купить новый телефон...` (러시아어)
  - `#Person1#은 #Person2#의불편함과 scratching에해perturbadoestá...` (스페인어)

**원인**:
- SOLAR-10.7B-Instruct는 영어 중심 사전학습 모델로, 1 epoch 파인튜닝만으로는 영어 생성 편향을 충분히 극복하지 못함
- 모델이 다양한 언어로 사전학습되어 있어, 입력 맥락에 따라 러시아어·스페인어가 나오기도 함
- `is_korean()` 단순 체크(한글 유무)로는 혼합 케이스를 포착하지 못함

**초기 감지 로직 문제**:
```python
# 잘못된 방식: 한글 1글자만 있어도 통과
def is_korean(text): return bool(re.search(r'[가-힣]', str(text)))
```

**개선된 감지 로직**:
```python
# 한글 비율 30% 미만이면 재추론 대상
def is_korean(text):
    t = str(text)
    total = len(re.sub(r"[\s#\d\.\,\!\?\:\;\(\)\-\']", "", t))
    if total == 0: return True
    return len(re.findall(r"[가-힣]", t)) / total >= 0.3
```

**조치**:
- `fix_english_outputs.py` 작성: 비한국어 샘플 탐지 후 강화된 프롬프트로 재추론
- 강화 프롬프트: `[중요] 반드시 한국어로만 작성하세요. 영어 사용 금지. MUST respond in Korean only.`
- 원본 파일 백업: `solar-r16-lr2e4-ep1-newprompt_before_fix.csv`
- 재추론 대상: **309개** (한글 비율 30% 미만 전체 포함)

### 현재 상태 (2026-03-27)

- `fix_english_outputs.py` PID 671891 실행 중
- 속도: ~50초/샘플 × 309개 = **예상 완료 ~4시간 20분**
- 완료 후 `outputs/predictions/solar-r16-lr2e4-ep1-newprompt.csv` 덮어쓰기 (백업 있음)

---

## [2026-03-28] solar-r32-lr1e4-ep3-v2 학습 중단 및 재시작

### 문제: 학습 프로세스가 step 444/2337(18%)에서 에러 없이 종료

**현황**:
- `EXP-20260327-193647` 실험 (어제 19:36 시작) → 22:48경 로그 중단
- GPU 완전 유휴 상태 (`3MiB` 사용, 프로세스 없음)
- 로그에 에러 메시지 없음 (OOM, CUDA error 등 없음)
- 체크포인트 디렉터리 **완전히 비어 있음** → 복구 불가

**원인**:
- 외부 요인으로 프로세스 종료 추정 (서버 세션 끊김, OOM killer 등)
- `save_strategy="epoch"` 설정으로 에포크 완료 전 중단 시 체크포인트 미저장

**조치**:
- `trainer.py` save 설정 변경: `save_strategy="epoch"` → `save_strategy="steps"`, `save_steps=200`, `save_total_limit=2`
- 처음부터 재시작: `EXP-20260328-125900` (`solar-r32-lr0.0001-ep3`)
- 이후 200 스텝마다 체크포인트 저장, 최근 2개 유지 → 중간 중단 시 이어서 재시작 가능
