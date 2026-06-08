from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"

    # Slack Incoming Webhook (메시지 전송 전용, ts 추적 불가)
    slack_webhook_url: str = ""
    # Slack Bot Token (chat.postMessage + chat.update 지원, ts 추적 가능)
    slack_bot_token: str = ""
    slack_channel_id: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./agent.db"

    repos_path: str = "/app/repos"

    # ChromaDB (서버 모드)
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # Ollama embedding model for RAG
    ollama_embed_model: str = "nomic-embed-text"

    # Public URL of this backend (for Slack approve/reject links)
    public_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:5173"


settings = Settings()
