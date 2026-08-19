"""
Configuration management for the Tdqeq pipeline.
Uses pydantic-settings to read from environment variables or .env files.
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TDQEQ_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
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




# Global settings instance
settings = Settings()
