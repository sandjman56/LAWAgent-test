from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- OpenAI ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embeddings_model: str = "text-embedding-3-large"

    # --- Perplexity ---
    perplexity_api_key: str | None = None
    perplexity_model: str = "sonar"  # or "llama-3.1-sonar-large-128k-online"

    # --- Microsoft Graph ---
    ms_client_id: str | None = None
    ms_client_secret: str | None = None
    ms_tenant_id: str | None = None

    # --- Limits ---
    max_file_mb: int = 15
    max_pages: int = 100

    # --- Server / CORS ---
    allowed_origins: str | None = None  # comma-separated
    port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"
        extra = "ignore"  # don’t crash on unused keys


settings = Settings()
