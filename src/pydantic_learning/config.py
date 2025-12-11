"""Central configuration for PydanticAI Learning Hub."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai.settings import ModelSettings


class AgentSettings(BaseSettings):
    """Central configuration for all PydanticAI agents.

    All models default to openai:gpt-5-nano for consistency.
    Override via environment variables with AGENT_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix='AGENT_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # All models default to gpt-5-nano
    default_model: str = Field(
        default='openai:gpt-5-nano',
        description='Default model for general agents'
    )
    answerer_model: str = Field(
        default='openai:gpt-5-nano',
        description='Model for answerer/responder agents'
    )
    questioner_model: str = Field(
        default='openai:gpt-5-nano',
        description='Model for questioner/planner agents'
    )
    search_model: str = Field(
        default='openai:gpt-5-nano',
        description='Model for search/web agents'
    )
    analysis_model: str = Field(
        default='openai:gpt-5-nano',
        description='Model for analysis/synthesis agents'
    )

    # Model settings
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    timeout: float = Field(default=60.0, ge=1.0)

    def to_model_settings(self) -> ModelSettings:
        """Convert to PydanticAI ModelSettings for use with agents.

        This helper method creates a ModelSettings instance from the configured
        values, making it easy to apply consistent settings across all agents.

        Returns:
            ModelSettings: Configured model settings for PydanticAI agents.

        Example:
            >>> from pydantic_learning.config import settings
            >>> agent = Agent(
            ...     settings.default_model,
            ...     instructions='...',
            ...     model_settings=settings.to_model_settings(),
            ... )
        """
        return ModelSettings(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )


class InfraSettings(BaseSettings):
    """Infrastructure configuration for durable execution."""

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore',
    )

    temporal_host: str = Field(default='localhost:7233')
    temporal_task_queue: str = Field(default='pydantic-learning')
    dbos_database_url: str = Field(
        default='postgresql://postgres@localhost:5432/dbos'
    )
    logfire_enabled: bool = Field(default=True)
    logfire_console: bool = Field(default=False)


@lru_cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()


@lru_cache
def get_infra_settings() -> InfraSettings:
    return InfraSettings()


# Backward compatibility - expose singleton
settings = get_agent_settings()
