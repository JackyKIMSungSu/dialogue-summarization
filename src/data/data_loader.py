from pathlib import Path

import pandas as pd
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import DialogueSummarizationDataset
from .preprocessor import DialoguePreprocessor


def load_config(config_path: str = "configs/base_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataloaders(
    config: dict,
    tokenizer: AutoTokenizer,
    prompt_template: str | None = None,
    model_type: str = "seq2seq",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """train / dev / test DataLoader를 생성합니다.

    Args:
        config:          base_config.yaml에서 로드한 설정 딕셔너리.
        tokenizer:       HuggingFace tokenizer.
        prompt_template: 입력 프롬프트 포맷. `{dialogue}` 플레이스홀더 포함.
        model_type:      "seq2seq" | "causal"

    Returns:
        (train_loader, dev_loader, test_loader)
    """
    data_cfg  = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    preprocessor = DialoguePreprocessor(
        prompt_template=prompt_template,
        max_dialogue_len=None,
    )

    train_df = preprocessor.process(pd.read_csv(data_cfg["train_path"]))
    dev_df   = preprocessor.process(pd.read_csv(data_cfg["dev_path"]))
    test_df  = preprocessor.process(pd.read_csv(data_cfg["test_path"]))

    dataset_kwargs = dict(
        tokenizer=tokenizer,
        max_input_len=model_cfg["max_input_length"],
        max_target_len=model_cfg["max_output_length"],
        model_type=model_type,
    )

    train_ds = DialogueSummarizationDataset(train_df, target_col=data_cfg["summary_col"], **dataset_kwargs)
    dev_ds   = DialogueSummarizationDataset(dev_df,   target_col=data_cfg["summary_col"], **dataset_kwargs)
    test_ds  = DialogueSummarizationDataset(test_df,  target_col=None,                   **dataset_kwargs)

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],      shuffle=True,  num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=train_cfg["eval_batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg["eval_batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    print(f"train: {len(train_ds):,}  |  dev: {len(dev_ds):,}  |  test: {len(test_ds):,}")
    return train_loader, dev_loader, test_loader
