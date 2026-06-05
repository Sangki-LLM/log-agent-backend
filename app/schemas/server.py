from pydantic import BaseModel


class ServerCreate(BaseModel):
    name: str
    hosts: list[str]
    git_repo_url: str
    git_branch: str = "main"
    github_token: str = ""


class ServerResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    hosts: list[str]
    git_repo_url: str
    git_branch: str
    is_active: bool

    @classmethod
    def from_orm_with_hosts(cls, server) -> "ServerResponse":
        return cls(
            id=server.id,
            name=server.name,
            hosts=[h.host for h in server.hosts],
            git_repo_url=server.git_repo_url,
            git_branch=server.git_branch,
            is_active=server.is_active,
        )


class ErrorEventPayload(BaseModel):
    server_name: str
    server_ip: str
    error_type: str = ""
    message: str = ""
    stack_trace: str = ""
    request_method: str = ""
    request_url: str = ""
    request_body: str = ""
    response_status: int = 500


