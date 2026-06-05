from pydantic import BaseModel


class ServerCreate(BaseModel):
    name: str
    host: str  # IP 주소


class ServerResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    host: str
    is_active: bool


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
