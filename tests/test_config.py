"""Tests for configuration modules."""
import sys
from pathlib import Path
import pytest
from pydantic_ai.settings import ModelSettings

# Add pydantic-ai-learning to path for imports
LEARNING_PATH = Path(__file__).parent.parent / "pydantic-ai-learning"
if str(LEARNING_PATH) not in sys.path:
    sys.path.insert(0, str(LEARNING_PATH))


class TestAppSettings:
    """Tests for pydantic-ai-learning AppSettings configuration."""

    def test_default_values(self):
        """Settings should have sensible defaults."""
        from backend.app.settings import AppSettings

        settings = AppSettings()
        assert settings.default_model == "openai:gpt-4o"
        assert settings.temporal_host == "localhost:7233"
        assert settings.task_queue == "learning-tasks"
        assert settings.logfire_enabled is False

    def test_model_settings(self):
        """get_model_settings should return proper ModelSettings."""
        from backend.app.settings import AppSettings

        settings = AppSettings()
        model_settings = settings.get_model_settings()

        assert isinstance(model_settings, ModelSettings)
        assert model_settings.temperature == 0.7
        assert model_settings.timeout == 30.0

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("APP_DEFAULT_MODEL", "anthropic:claude-sonnet-4-5")
        monkeypatch.setenv("APP_TEMPORAL_HOST", "custom-host:7233")
        monkeypatch.setenv("APP_LOGFIRE_ENABLED", "true")

        from backend.app.settings import AppSettings

        settings = AppSettings()
        assert settings.default_model == "anthropic:claude-sonnet-4-5"
        assert settings.temporal_host == "custom-host:7233"
        assert settings.logfire_enabled is True


class TestAgentSettings:
    """Tests for agent configuration best practices."""

    def test_model_settings_creation(self):
        """ModelSettings should be creatable with custom values."""
        model_settings = ModelSettings(
            temperature=0.5,
            max_tokens=1000,
            timeout=60.0
        )

        assert model_settings.temperature == 0.5
        assert model_settings.max_tokens == 1000
        assert model_settings.timeout == 60.0

    def test_model_settings_defaults(self):
        """ModelSettings should work with defaults."""
        model_settings = ModelSettings()

        # ModelSettings has reasonable defaults
        assert model_settings.temperature is None or isinstance(model_settings.temperature, (int, float))
        assert model_settings.timeout is None or isinstance(model_settings.timeout, (int, float))
