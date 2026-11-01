from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = ""
    gemini_timeout_seconds: float = 45.0
    gemini_max_attempts: int = 3
    gemini_backoff_base_seconds: float = 1.0
    gemini_temperature: float = 0.1
    gemini_price_input_per_million_usd: float = 0.30
    gemini_price_output_per_million_usd: float = 2.50
    gemini_cache_size: int = 1000

    ml_confidence_threshold: float = 0.65
    max_text_length: int = 1000
    max_batch_size: int = 50

    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"
    reports_dir: Path = BASE_DIR / "reports"
    database_path: Path = BASE_DIR / "data" / "chamados.db"

    cors_origins: str = "*"
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
