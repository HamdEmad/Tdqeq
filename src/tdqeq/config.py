"""
Configuration management for the Tdqeq pipeline.
Uses pydantic-settings to read from environment variables or .env files.
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TDQEQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Model Weights
    YOLO_WEIGHTS_PATH: Optional[str] = None  # If None, it will auto-download
    TABLE_CLS_WEIGHTS_PATH: Optional[str] = None  # If None, it will auto-download
    TABLE_UNET_WEIGHTS_PATH: Optional[str] = None  # If None, it will auto-download

    # Logging
    LOG_LEVEL: str = "INFO"

    # PDF Rendering
    DEFAULT_DPI: int = 150

    # Inference Options
    DEFAULT_BATCH_SIZE: int = 4

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = "sk-MuVZaTUtHt7Vnim3wDL71Q"
    OPENAI_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    OPENAI_BASE_URL: Optional[str] = "http://10.203.12.137:4000"


# Global settings instance
settings = Settings()

