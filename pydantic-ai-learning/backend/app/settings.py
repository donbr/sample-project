from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai.settings import ModelSettings

class AppSettings(BaseSettings):
    """Configuration for the PydanticAI Learning App."""
    
    model_config = SettingsConfigDict(
        env_prefix='APP_',
        env_file='.env',
        extra='ignore'
    )

    # Model Configuration
    openai_api_key: str | None = Field(default=None, alias='OPENAI_API_KEY')
    default_model: str = Field(default='openai:gpt-4o')
    
    # Temporal Configuration
    temporal_host: str = "localhost:7233"
    task_queue: str = "learning-tasks"
    
    # Observability
    logfire_enabled: bool = False

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(
            temperature=0.7,
            timeout=30.0
        )

# Singleton to be used across the app
settings = AppSettings()
