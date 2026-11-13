from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://ilm:ilm@localhost:5432/ilm"
    cors_origins: str = "http://localhost:5173"
    api_prefix: str = "/api/v1"

    admin_enabled: bool = True
    admin_username: str = "admin"
    admin_password: str = ""
    session_secret: str = "change-me"

    upload_dir: str = "./storage/uploads"

    cohere_api_key: str = ""
    embedding_model: str = "embed-multilingual-v3.0"
    embed_batch_size: int = 96
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    generate_followups: bool = False
    similarity_warning: float = 0.78
    rerank_enabled: bool = False
    rerank_model: str = "rerank-multilingual-v3.0"
    rerank_candidates: int = 50
    sql_echo: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

settings = Settings()
