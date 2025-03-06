from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized application configuration.
    Environment variables are loaded automatically from a .env file.
    """

    # LLM configuration
    LLM_BASE_URL: str = Field(..., description="Base URL for the LLM service")
    LLM_MODEL: str = Field(..., description="Model name for the LLM service")
    LLM_API_KEY: SecretStr = Field(..., description="API key for the LLM service")

    # STT configuration
    STT_BASE_URL: str = Field(..., description="Base URL for the STT service")
    STT_API_KEY: SecretStr = Field("", description="Optional API key for the STT service")
    STT_MODEL: str = Field(..., description="Model name for the STT service")
    STT_RESPONSE_FORMAT: str = Field(..., description="Response format for the STT service")
    LANGUAGE: str = Field("en", description="Language setting for the STT service")

    # TTS configuration
    TTS_BASE_URL: str = Field(..., description="Base URL for the TTS service")
    TTS_API_KEY: SecretStr = Field(..., description="API key for the TTS service")
    TTS_MODEL: str = Field(..., description="Model name for the TTS service")
    TTS_VOICE: str = Field(..., description="Voice(s) to use for the TTS service")
    TTS_BACKEND: str = Field(..., description="Backend identifier for TTS processing")
    TTS_AUDIO_FORMAT: str = Field(..., description="Audio format for TTS service")

    # Application mode
    MODE: str = Field(..., description="Mode of the application, e.g. 'UI', 'PHONE'")

    # Load environment variables from .env file using UTF-8 encoding.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Optionally, for use in web frameworks, wrap settings retrieval in a function with caching:
from functools import lru_cache

@lru_cache()
def get_settings() -> Settings:
    return Settings()
