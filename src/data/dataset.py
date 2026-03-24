from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DialogueSummarizationDataset(Dataset):
    """대화 요약 PyTorch Dataset.

    Encoder-Decoder (seq2seq) 모델용:
        input_ids, attention_mask → dialogue 토크나이즈
        labels                   → summary 토크나이즈

    Decoder-Only (LLM) 모델용:
        input_ids, attention_mask → prompt + summary 토크나이즈 (concat)
        labels                   → prompt 부분은 -100 마스킹
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        input_col: str = "input_text",
        target_col: str | None = "summary",
        max_input_len: int = 1024,
        max_target_len: int = 256,
        model_type: str = "seq2seq",  # "seq2seq" | "causal"
    ):
        """
        Args:
            df:             전처리된 DataFrame.
            tokenizer:      HuggingFace tokenizer.
            input_col:      입력 텍스트 컬럼명.
            target_col:     정답 요약 컬럼명. None이면 inference 전용.
            max_input_len:  입력 최대 토큰 수.
            max_target_len: 출력 최대 토큰 수.
            model_type:     "seq2seq" 또는 "causal".
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.input_col = input_col
        self.target_col = target_col
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.model_type = model_type

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        input_text = row[self.input_col]
        target_text = row[self.target_col] if self.target_col else None

        if self.model_type == "seq2seq":
            return self._encode_seq2seq(input_text, target_text)
        else:
            return self._encode_causal(input_text, target_text)

    # ------------------------------------------------------------------
    # Seq2Seq 인코딩
    # ------------------------------------------------------------------

    def _encode_seq2seq(self, input_text: str, target_text: str | None) -> dict:
        enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if target_text is not None:
            with self.tokenizer.as_target_tokenizer():
                dec = self.tokenizer(
                    target_text,
                    max_length=self.max_target_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
            labels = dec["input_ids"].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            item["labels"] = labels

        return item

    # ------------------------------------------------------------------
    # Causal LM 인코딩
    # ------------------------------------------------------------------

    def _encode_causal(self, input_text: str, target_text: str | None) -> dict:
        if target_text is None:
            enc = self.tokenizer(
                input_text,
                max_length=self.max_input_len,
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }

        full_text = input_text + target_text
        prompt_ids = self.tokenizer(input_text, add_special_tokens=False)["input_ids"]
        enc = self.tokenizer(
            full_text,
            max_length=self.max_input_len + self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100                         # 프롬프트 마스킹
        labels[attention_mask == 0] = -100                       # 패딩 마스킹

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
