from datetime import datetime

from pydantic import BaseModel


class AnalysisRecordResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    server_id: int
    trigger_line: str
    raw_log: str
    llm_suggestion: str | None
    status: str
    slack_ts: str | None
    github_pr_url: str | None
    created_at: datetime
