"""Environment-backed backend configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="STREAMIQ_",
        extra="ignore",
    )

    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "http://localhost:5173"
    processed_video_root: Path = Path("video_assets/processed")
    experiment_root: Path = Path("experiments/results/streaming")
    dqn_checkpoint: Path = Path("experiments/results/dqn_sample/model_final.pt")
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
