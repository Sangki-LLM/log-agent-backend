from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"

    # Slack Incoming Webhook
    slack_webhook_url: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./agent.db"

    # Public URL of this backend (for Slack approve/reject links)
    public_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:5173"


settings = Settings()
