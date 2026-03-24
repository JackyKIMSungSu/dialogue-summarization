import re
import pandas as pd


class DialoguePreprocessor:
    """대화 요약 데이터 전처리 클래스.

    주요 처리:
    1. 공백/개행 정규화
    2. #PersonN# 태그 정규화 (선택)
    3. 길이 필터링 (선택)
    4. 모델 입력용 프롬프트 포맷 변환
    """

    SPEAKER_TAG_RE = re.compile(r'#Person(\d+)#')
    MULTI_SPACE_RE = re.compile(r'[ \t]+')
    MULTI_NEWLINE_RE = re.compile(r'\n{2,}')

    def __init__(
        self,
        prompt_template: str | None = None,
        anonymize_speakers: bool = False,
        max_dialogue_len: int | None = None,
    ):
        """
        Args:
            prompt_template:   모델 입력 포맷. `{dialogue}` 플레이스홀더 포함.
                               None이면 대화문 그대로 사용.
            anonymize_speakers: True면 #PersonN# → '화자N'으로 변환.
            max_dialogue_len:  이 길이(문자 수) 초과 시 해당 샘플을 필터링.
                               None이면 필터링 없음.
        """
        self.prompt_template = prompt_template or "{dialogue}"
        self.anonymize_speakers = anonymize_speakers
        self.max_dialogue_len = max_dialogue_len

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 전체를 전처리하여 반환합니다.

        Args:
            df: 원본 DataFrame. 'dialogue' 컬럼 필수.
                'summary' 컬럼이 있으면 함께 처리.

        Returns:
            전처리된 DataFrame (원본 수정 없음).
        """
        df = df.copy()

        df['dialogue'] = df['dialogue'].apply(self._clean_dialogue)

        if 'summary' in df.columns:
            df['summary'] = df['summary'].apply(self._clean_summary)

        if self.max_dialogue_len is not None:
            before = len(df)
            df = df[df['dialogue'].str.len() <= self.max_dialogue_len].reset_index(drop=True)
            print(f"[filter] {before - len(df)}개 제거 (max_len={self.max_dialogue_len}) → {len(df)}개 남음")

        df['input_text'] = df['dialogue'].apply(self._build_prompt)

        return df

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _clean_dialogue(self, text: str) -> str:
        text = self.MULTI_SPACE_RE.sub(' ', text)       # 연속 공백 → 단일 공백
        text = self.MULTI_NEWLINE_RE.sub('\n', text)    # 연속 개행 → 단일 개행
        text = text.strip()
        if self.anonymize_speakers:
            text = self.SPEAKER_TAG_RE.sub(lambda m: f'화자{m.group(1)}', text)
        return text

    def _clean_summary(self, text: str) -> str:
        text = self.MULTI_SPACE_RE.sub(' ', text)
        text = text.replace('\n', ' ')                  # 개행 → 공백
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _build_prompt(self, dialogue: str) -> str:
        return self.prompt_template.format(dialogue=dialogue)
