import os

import wandb
from dotenv import load_dotenv

load_dotenv()


class WandbLogger:
    """Weights & Biases 실험 로깅 유틸리티"""

    def __init__(self, config: dict):
        """
        Args:
            config: base_config.yaml에서 로드한 설정 딕셔너리
                    wandb 키 아래에 project, entity, name, tags 등을 정의
        """
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            try:
                wandb.login(key=api_key)
            except ValueError:
                # 새 키 형식(wandb_v1_...)은 구버전 wandb에서 지원 안 함
                # 기존 로그인 세션을 그대로 사용
                wandb.login()

        wandb_cfg = config.get("wandb", {})

        self.run = wandb.init(
            project=wandb_cfg.get("project", "dialogue-summarization"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("name", None),
            tags=wandb_cfg.get("tags", []),
            config=config,
            resume=wandb_cfg.get("resume", "allow"),
        )

    # ------------------------------------------------------------------
    # 학습 중 호출
    # ------------------------------------------------------------------

    def log(self, metrics: dict, step: int | None = None):
        """스칼라 지표 로깅 (loss, rouge 등)"""
        wandb.log(metrics, step=step)

    def log_summary(self, key: str, value):
        """run 요약(best score 등) 업데이트"""
        wandb.run.summary[key] = value

    def log_artifact(self, path: str, name: str, artifact_type: str = "model"):
        """파일/디렉터리를 Artifact로 업로드"""
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        wandb.log_artifact(artifact)

    def log_table(self, key: str, columns: list[str], data: list[list]):
        """예측 결과 테이블 로깅"""
        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table})

    # ------------------------------------------------------------------
    # 종료
    # ------------------------------------------------------------------

    def finish(self):
        wandb.finish()

    # ------------------------------------------------------------------
    # context manager 지원
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
