"""Application configuration for StreamIQ backend."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "StreamIQ API"
    app_version: str = "0.2.0"
    debug: bool = False
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ]
    model_checkpoint: Path = PROJECT_ROOT / "experiments" / "results" / "dqn" / "model_final.pt"
    video_assets_dir: Path = PROJECT_ROOT / "backend" / "assets" / "dash"

    model_config = SettingsConfigDict(env_prefix="STREAMIQ_")


settings = Settings()

