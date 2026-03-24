import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from notion_client import Client

load_dotenv()


class NotionLogger:
    """Notion 데이터베이스(Training Datasheet)에 실험 결과를 기록하는 유틸리티.

    DB 컬럼 구성:
        - 실험명        (title)
        - 상태          (status)   : 계획 / 진행 중 / 완료 / 실패
        - Run Name      (rich_text)
        - Experiment ID (rich_text)
        - Dataset       (rich_text)
        - Task          (select)   : Classification / NER / Summarization
        - rouge1        (number)
        - rouge2        (number)
        - rougeL        (number)
        - final_result  (number)   : rouge1 + rouge2 + rougeL
        - 목적          (rich_text)
        - 결과 요약     (rich_text)
        - 실험일        (date)
        - 태그          (multi_select)
    """

    def __init__(self, token: str | None = None, database_id: str | None = None):
        """
        Args:
            token:       Notion Integration Token.
                         None이면 환경변수 NOTION_TOKEN 사용.
            database_id: 기록할 Notion DB ID.
                         None이면 환경변수 NOTION_DATABASE_ID 사용.
        """
        self.client = Client(auth=token or os.environ["NOTION_TOKEN"])
        self.database_id = database_id or os.environ["NOTION_DATABASE_ID"]
        self._page_id: str | None = None

    # ------------------------------------------------------------------
    # 페이지 생성 / 업데이트
    # ------------------------------------------------------------------

    def create_run(
        self,
        실험명: str,
        run_name: str = "",
        experiment_id: str = "",
        dataset: str = "",
        task: str = "Summarization",
        목적: str = "",
        태그: list[str] | None = None,
    ) -> str:
        """실험 시작 시 Notion DB에 새 행(페이지)을 생성합니다.

        Returns:
            생성된 페이지 ID (이후 update_run에 전달)
        """
        props: dict = {
            "실험명": {"title": [{"text": {"content": 실험명}}]},
            "상태": {"status": {"name": "진행 중"}},
            "실험일": {"date": {"start": datetime.now(timezone.utc).date().isoformat()}},
        }
        if run_name:
            props["Run Name"] = {"rich_text": [{"text": {"content": run_name}}]}
        if experiment_id:
            props["Experiment ID"] = {"rich_text": [{"text": {"content": experiment_id}}]}
        if dataset:
            props["Dataset"] = {"rich_text": [{"text": {"content": dataset}}]}
        if task:
            props["Task"] = {"select": {"name": task}}
        if 목적:
            props["목적"] = {"rich_text": [{"text": {"content": 목적}}]}
        if 태그:
            props["태그"] = {"multi_select": [{"name": t} for t in 태그]}

        response = self.client.pages.create(
            parent={"database_id": self.database_id},
            properties=props,
        )
        self._page_id = response["id"]
        return self._page_id

    def update_run(
        self,
        rouge1: float = 0.0,
        rouge2: float = 0.0,
        rougeL: float = 0.0,
        결과_요약: str = "",
        상태: str = "완료",
        page_id: str | None = None,
    ):
        """실험 종료 후 ROUGE 점수 및 상태를 업데이트합니다."""
        pid = page_id or self._page_id
        if pid is None:
            raise ValueError("page_id가 없습니다. create_run()을 먼저 호출하세요.")

        props = {
            "상태": {"status": {"name": 상태}},
            "rouge1": {"number": round(rouge1, 4)},
            "rouge2": {"number": round(rouge2, 4)},
            "rougeL": {"number": round(rougeL, 4)},
            "final_result": {"number": round(rouge1 + rouge2 + rougeL, 4)},
        }
        if 결과_요약:
            props["결과 요약"] = {"rich_text": [{"text": {"content": 결과_요약}}]}

        self.client.pages.update(page_id=pid, properties=props)

    def fail_run(self, error_msg: str = "", page_id: str | None = None):
        """예외 발생 시 상태를 실패로 표시합니다."""
        pid = page_id or self._page_id
        if pid is None:
            return
        props = {"상태": {"status": {"name": "실패"}}}
        if error_msg:
            props["결과 요약"] = {"rich_text": [{"text": {"content": error_msg[:2000]}}]}
        self.client.pages.update(page_id=pid, properties=props)

    # ------------------------------------------------------------------
    # context manager 지원
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.fail_run(error_msg=str(exc_val))
        return False
