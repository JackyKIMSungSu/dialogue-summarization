"""
제출 파일 생성 모듈.

사용 예시:
    from src.inference.submit import SubmissionGenerator

    gen = SubmissionGenerator()
    gen.save(fnames=test_df['fname'].tolist(), predictions=predictions)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd


class SubmissionGenerator:
    """대회 제출용 CSV 파일 생성기.

    제출 형식:
        Unnamed: 0  fname    summary
        0           test_0   요약문
        1           test_1   요약문
        ...
    """

    def __init__(
        self,
        sample_submission_path: str = "data/raw/sample_submission.csv",
        output_dir: str = "outputs/predictions",
    ):
        """
        Args:
            sample_submission_path: 대회에서 제공된 sample_submission.csv 경로.
            output_dir:             제출 파일 저장 디렉터리.
        """
        self.sample_path = Path(sample_submission_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._template = pd.read_csv(self.sample_path)

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def save(
        self,
        predictions: list[str],
        fnames: list[str] | None = None,
        filename: str | None = None,
    ) -> Path:
        """예측 결과를 제출 형식의 CSV로 저장합니다.

        Args:
            predictions: 모델이 생성한 요약문 리스트 (test 순서와 동일).
            fnames:      fname 리스트. None이면 sample_submission의 fname 사용.
            filename:    저장 파일명. None이면 타임스탬프로 자동 생성.

        Returns:
            저장된 파일 경로.
        """
        template = self._template.copy()

        if fnames is not None:
            assert len(fnames) == len(template), \
                f"fnames 길이({len(fnames)})와 template 길이({len(template)})가 다릅니다."
            template["fname"] = fnames

        assert len(predictions) == len(template), \
            f"predictions 길이({len(predictions)})와 template 길이({len(template)})가 다릅니다."

        template["summary"] = [self._clean(p) for p in predictions]

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{timestamp}.csv"

        save_path = self.output_dir / filename
        template.to_csv(save_path, index=False)

        print(f"[submission] {len(template)}개 저장 완료 → {save_path}")
        self._validate(template)
        return save_path

    def from_df(
        self,
        df: pd.DataFrame,
        pred_col: str = "prediction",
        fname_col: str = "fname",
        filename: str | None = None,
    ) -> Path:
        """DataFrame에서 직접 제출 파일을 생성합니다.

        Args:
            df:        fname과 prediction 컬럼이 포함된 DataFrame.
            pred_col:  예측 요약 컬럼명.
            fname_col: fname 컬럼명.
            filename:  저장 파일명.
        """
        return self.save(
            predictions=df[pred_col].tolist(),
            fnames=df[fname_col].tolist() if fname_col in df.columns else None,
            filename=filename,
        )

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        """빈 예측값 처리 및 공백 정리."""
        text = str(text).strip()
        if not text:
            return "요약 없음"
        return text

    def _validate(self, df: pd.DataFrame):
        """제출 파일 기본 검증."""
        issues = []

        empty_mask = df["summary"].str.strip().eq("")
        if empty_mask.any():
            issues.append(f"빈 요약문 {empty_mask.sum()}개: {df.loc[empty_mask, 'fname'].tolist()[:5]}")

        dup_mask = df["fname"].duplicated()
        if dup_mask.any():
            issues.append(f"fname 중복 {dup_mask.sum()}개")

        expected_fnames = self._template["fname"].tolist()
        if df["fname"].tolist() != expected_fnames:
            issues.append("fname 순서가 sample_submission과 다릅니다.")

        if issues:
            print("[validation] 경고:")
            for msg in issues:
                print(f"  - {msg}")
        else:
            print("[validation] 이상 없음")
