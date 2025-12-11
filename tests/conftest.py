"""
Pytest configuration and fixtures for PydanticAI testing.

CRITICAL: All tests use TestModel to avoid real API calls.
"""
import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel


@pytest.fixture(autouse=True)
def block_real_model_requests():
    """
    CRITICAL: Prevent any real LLM API calls during tests.

    This fixture runs automatically for ALL tests.
    If a test accidentally tries to call a real model,
    it will raise an error instead of making an API call.
    """
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    yield
    models.ALLOW_MODEL_REQUESTS = original


@pytest.fixture
def test_model():
    """Provide a fresh TestModel instance."""
    return TestModel()


@pytest.fixture
def test_model_with_text():
    """TestModel that returns custom text responses."""
    def _factory(text: str) -> TestModel:
        return TestModel(custom_output_text=text)
    return _factory


@pytest.fixture
def test_model_with_structured():
    """TestModel that returns custom structured responses."""
    def _factory(**kwargs) -> TestModel:
        return TestModel(custom_output_args=kwargs)
    return _factory
