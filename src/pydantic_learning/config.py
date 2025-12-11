"""
Centralized Configuration for PydanticAI Learning Modules
==========================================================

This module provides centralized configuration for all learning examples.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai.settings import ModelSettings


class AgentSettings(BaseSettings):
    """Configuration for PydanticAI agents."""

    model_config = SettingsConfigDict(
        env_prefix='PYDANTIC_LEARNING_',
        env_file='.env',
        extra='ignore'
    )

    # Model Configuration
    openai_api_key: str | None = Field(default=None, alias='OPENAI_API_KEY')
    anthropic_api_key: str | None = Field(default=None, alias='ANTHROPIC_API_KEY')
    default_model: str = Field(default='openai:gpt-4o')

    # Temporal Configuration
    temporal_host: str = "localhost:7233"
    task_queue: str = "learning-tasks"

    # DBOS Configuration
    dbos_host: str = "localhost"
    dbos_port: int = 5432

    # Observability
    logfire_enabled: bool = False

    def get_model_settings(self) -> ModelSettings:
        """Get model settings for agent configuration."""
        return ModelSettings(
            temperature=0.7,
            timeout=30.0
        )


def get_agent_settings() -> AgentSettings:
    """Get the singleton agent settings instance."""
    return AgentSettings()
