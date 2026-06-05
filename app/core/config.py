from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"

    # Slack
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_channel_id: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./agent.db"

    # Security
    pem_encryption_key: str = ""

    # Agent backend public URL (for Slack webhook callbacks)
    public_url: str = "http://localhost:8000"


settings = Settings()
