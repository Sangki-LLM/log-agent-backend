from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"

    # Slack Incoming Webhook (전역 기본값, 서버별 설정이 없을 때 사용)
    slack_webhook_url: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./agent.db"

    repos_path: str = "/app/repos"

    # ChromaDB (서버 모드)
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # Ollama embedding model for RAG
    ollama_embed_model: str = "nomic-embed-text"

    # Gemini API (LLM Judge)
    gemini_api_key: str = ""

    # Public URL of this backend (for Slack approve/reject links)
    public_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:5173"


settings = Settings()
