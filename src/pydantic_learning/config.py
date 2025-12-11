"""Central configuration for PydanticAI agents using pydantic-settings.

This module provides environment-based configuration for all PydanticAI agents
and infrastructure components used in the project.

References:
    - docs/pydantic_ai_best_practices.md - Configuration Management section
    - https://docs.pydantic.dev/latest/concepts/pydantic_settings/
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Central configuration for all PydanticAI agents.

    This class manages model selection and settings for different agent roles.
    All settings can be overridden via environment variables with the AGENT_ prefix.

    Environment Variables:
        AGENT_DEFAULT_MODEL: Default model for general agents
        AGENT_ANSWERER_MODEL: Model for answerer/responder agents
        AGENT_QUESTIONER_MODEL: Model for questioner/planner agents
        AGENT_SEARCH_MODEL: Model for search/web agents
        AGENT_ANALYSIS_MODEL: Model for analysis/synthesis agents
        AGENT_TEMPERATURE: Model temperature (0.0-2.0)
        AGENT_MAX_TOKENS: Maximum tokens to generate
        AGENT_TIMEOUT: Request timeout in seconds

    Examples:
        >>> settings = get_agent_settings()
        >>> agent = Agent(settings.analysis_model, ...)

        # Override via environment:
        # AGENT_ANALYSIS_MODEL=anthropic:claude-opus-4-5 python my_app.py
    """

    model_config = SettingsConfigDict(
        env_prefix='AGENT_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # Default models for different agent roles
    # All default to gpt-5-nano for consistency, override via environment
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
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description='Randomness: 0.0 = deterministic, 2.0 = very creative'
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description='Maximum tokens to generate before stopping'
    )
    timeout: float = Field(
        default=60.0,
        ge=1.0,
        description='Request timeout in seconds'
    )


class InfraSettings(BaseSettings):
    """Infrastructure configuration for durable execution.

    This class manages configuration for infrastructure components like
    Temporal, DBOS (Postgres), and observability tools.

    Environment Variables:
        TEMPORAL_HOST: Temporal server host and port
        TEMPORAL_TASK_QUEUE: Temporal task queue name
        DBOS_DATABASE_URL: PostgreSQL connection URL for DBOS
        LOGFIRE_ENABLED: Enable Logfire observability
        LOGFIRE_CONSOLE: Enable Logfire console output

    Examples:
        >>> settings = get_infra_settings()
        >>> # Use settings.temporal_host for Temporal connection
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # Temporal
    temporal_host: str = Field(
        default='localhost:7233',
        description='Temporal server host and port'
    )
    temporal_task_queue: str = Field(
        default='pydantic-learning',
        description='Temporal task queue name'
    )

    # DBOS (Postgres)
    dbos_database_url: str = Field(
        default='postgresql://postgres@localhost:5432/dbos',
        description='PostgreSQL connection URL for DBOS'
    )

    # Observability
    logfire_enabled: bool = Field(
        default=True,
        description='Enable Logfire observability'
    )
    logfire_console: bool = Field(
        default=False,
        description='Enable Logfire console output'
    )


@lru_cache
def get_agent_settings() -> AgentSettings:
    """Get cached agent settings singleton.

    This function uses LRU cache to ensure settings are loaded only once
    and reused across the application.

    Returns:
        AgentSettings: Singleton instance of agent configuration

    Examples:
        >>> settings = get_agent_settings()
        >>> agent = Agent(settings.analysis_model, ...)
    """
    return AgentSettings()


@lru_cache
def get_infra_settings() -> InfraSettings:
    """Get cached infrastructure settings singleton.

    This function uses LRU cache to ensure settings are loaded only once
    and reused across the application.

    Returns:
        InfraSettings: Singleton instance of infrastructure configuration

    Examples:
        >>> settings = get_infra_settings()
        >>> # Connect to Temporal using settings.temporal_host
    """
    return InfraSettings()
