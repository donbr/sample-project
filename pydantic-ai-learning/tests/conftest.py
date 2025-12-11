import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

# BEST PRACTICE:
# Always block real model requests in tests to ensure they are deterministic,
# fast, and free.
@pytest.fixture(autouse=True)
def block_real_model_requests():
    """Prevent any real LLM API calls during tests."""
    # This ensures that if any code tries to make a real
    # network request to an LLM, it will raise a RuntimeError.
    original_value = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    try:
        yield
    finally:
        models.ALLOW_MODEL_REQUESTS = original_value

@pytest.fixture
def test_model():
    """A standard TestModel to use in overrides."""
    return TestModel()
