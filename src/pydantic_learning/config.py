"""Central configuration for PydanticAI Learning Hub.

This module uses pydantic-settings to manage configuration from environment variables.
All model configurations and API settings are centralized here.

Environment variables can be set in a .env file at the project root.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        basic_model: Model for basic examples (lessons 01-02)
        haiku_model: Claude Haiku model for fast, cost-effective operations
        sonnet_model: Claude Sonnet model for complex reasoning
        gpt4_model: GPT-4 model for comparison and multi-agent examples
        openrouter_api_key: Optional OpenRouter API key for gateway models
        anthropic_api_key: Optional Anthropic API key for direct Claude access
        openai_api_key: Optional OpenAI API key for direct GPT access
    """

    # Model configurations - defaults use gateway prefix for OpenRouter
    basic_model: str = 'anthropic:claude-3-5-haiku-latest'
    haiku_model: str = 'gateway/anthropic:claude-3-5-haiku-latest'
    sonnet_model: str = 'gateway/anthropic:claude-3-5-sonnet-latest'
    gpt4_model: str = 'gateway/openai:gpt-4.1'

    # API Keys (optional - will use environment defaults if not set)
    openrouter_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Logfire configuration
    logfire_token: str | None = None

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )


# Global settings instance
settings = Settings()
