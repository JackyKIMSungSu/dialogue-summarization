"""
대회 공식 평가 방식에 따른 ROUGE 평가 모듈.

평가 방식:
  - MeCab 형태소 분석기로 토크나이즈 후 ROUGE 계산
  - ROUGE-1-F1 + ROUGE-2-F1 + ROUGE-L-F1 합산을 최종 점수로 사용
"""

from __future__ import annotations

import warnings
from typing import Sequence

import MeCab
import pandas as pd
from rouge import Rouge


class RougeEvaluator:
    """MeCab 기반 한국어 ROUGE 평가기 (대회 공식 평가 방식)."""

    def __init__(self):
        self._tagger = MeCab.Tagger('-Owakati')
        self._rouge = Rouge()

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def score(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> dict[str, float]:
        """예측 요약과 정답 요약의 ROUGE 점수를 계산합니다.

        Args:
            predictions: 모델이 생성한 요약문 리스트.
            references:  정답 요약문 리스트.

        Returns:
            {
                "rouge1": float,
                "rouge2": float,
                "rougeL": float,
                "score":  float,   # rouge1 + rouge2 + rougeL (대회 최종 점수)
            }
        """
        assert len(predictions) == len(references), \
            f"predictions({len(predictions)})와 references({len(references)}) 길이가 다릅니다."

        preds_tok = [self._tokenize(p) for p in predictions]
        refs_tok  = [self._tokenize(r) for r in references]

        valid_pairs = [
            (p, r) for p, r in zip(preds_tok, refs_tok)
            if p.strip() and r.strip()
        ]
        skipped = len(predictions) - len(valid_pairs)
        if skipped:
            warnings.warn(f"{skipped}개 샘플이 빈 문자열이어서 평가에서 제외됩니다.")

        if not valid_pairs:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "score": 0.0}

        preds_valid, refs_valid = zip(*valid_pairs)
        raw = self._rouge.get_scores(list(preds_valid), list(refs_valid), avg=True)

        rouge1 = raw["rouge-1"]["f"]
        rouge2 = raw["rouge-2"]["f"]
        rougeL = raw["rouge-l"]["f"]

        return {
            "rouge1": round(rouge1, 4),
            "rouge2": round(rouge2, 4),
            "rougeL": round(rougeL, 4),
            "score":  round(rouge1 + rouge2 + rougeL, 4),
        }

    def score_sample(self, prediction: str, reference: str) -> dict[str, float]:
        """단일 샘플의 ROUGE 점수를 계산합니다."""
        return self.score([prediction], [reference])

    def evaluate_df(
        self,
        df: pd.DataFrame,
        pred_col: str = "prediction",
        ref_col: str = "summary",
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """DataFrame에 샘플별 ROUGE를 추가하고 평균 점수를 반환합니다.

        Args:
            df:       예측값과 정답이 담긴 DataFrame.
            pred_col: 예측 요약 컬럼명.
            ref_col:  정답 요약 컬럼명.

        Returns:
            (avg_scores, df_with_per_sample_scores)
            avg_scores: {"rouge1", "rouge2", "rougeL", "score"}
            df: 원본 + rouge1/rouge2/rougeL/score 컬럼 추가
        """
        df = df.copy()
        per_sample = []
        for pred, ref in zip(df[pred_col], df[ref_col]):
            per_sample.append(self.score_sample(str(pred), str(ref)))

        df["rouge1"] = [s["rouge1"] for s in per_sample]
        df["rouge2"] = [s["rouge2"] for s in per_sample]
        df["rougeL"] = [s["rougeL"] for s in per_sample]
        df["score"]  = [s["score"]  for s in per_sample]

        avg = self.score(
            df[pred_col].astype(str).tolist(),
            df[ref_col].astype(str).tolist(),
        )
        return avg, df

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> str:
        """MeCab으로 형태소 분석 후 공백으로 이어붙인 문자열을 반환합니다."""
        text = str(text).strip()
        if not text:
            return ""
        return self._tagger.parse(text).strip()
