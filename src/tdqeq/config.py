"""
Configuration management for the Tdqeq pipeline.
Uses pydantic-settings to read from environment variables or .env files.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model Weights
    YOLO_WEIGHTS_PATH: Optional[str] = None  # If None, it will auto-download

    # Logging
    LOG_LEVEL: str = "INFO"

    # PDF Rendering
    DEFAULT_DPI: int = 150

    # Inference Options
    DEFAULT_BATCH_SIZE: int = 4

    class Config:
        env_prefix = "TDQEQ_"


# Global settings instance
settings = Settings()
