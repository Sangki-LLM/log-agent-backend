from pydantic import BaseModel


class ServerCreate(BaseModel):
    name: str
    host: str
    username: str = "ec2-user"
    pem_key: str
    project_path: str
    log_path: str
    git_branch: str = "main"


class ServerResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    host: str
    username: str
    project_path: str
    log_path: str
    git_branch: str
    is_active: bool


class ErrorEventPayload(BaseModel):
    server_id: int
    trigger_line: str
    stack_trace: str = ""
    context_b64: str = ""
    request_path: str = ""
