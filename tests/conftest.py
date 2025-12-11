"""Pytest configuration and fixtures for PydanticAI Learning Hub tests.

This module provides fixtures for testing agents without making actual API calls.
The ALLOW_MODEL_REQUESTS environment variable controls whether real API calls are allowed.
"""

import os

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


@pytest.fixture
def allow_model_requests() -> bool:
    """Determine if real model requests are allowed during testing.

    Returns:
        bool: True if ALLOW_MODEL_REQUESTS is set to 'true', False otherwise
    """
    return os.getenv('ALLOW_MODEL_REQUESTS', 'false').lower() == 'true'


@pytest.fixture
def test_model() -> TestModel:
    """Provide a TestModel for testing without API calls.

    The TestModel is a mock model that simulates LLM responses without making
    actual API calls. This is useful for unit testing agent logic.

    Returns:
        TestModel: A configured test model instance
    """
    return TestModel()


@pytest.fixture
def basic_test_agent(test_model: TestModel) -> Agent:
    """Provide a basic agent configured with TestModel.

    Args:
        test_model: The test model fixture

    Returns:
        Agent: A basic agent using the test model
    """
    return Agent(
        test_model,
        instructions='You are a helpful assistant for testing.',
    )


@pytest.fixture
def agent_with_custom_response(test_model: TestModel) -> callable:
    """Factory fixture for creating agents with custom test responses.

    Returns:
        callable: A function that creates an agent with a predefined response

    Example:
        ```python
        def test_something(agent_with_custom_response):
            agent = agent_with_custom_response("Custom response text")
            result = await agent.run("Test query")
            assert result.output == "Custom response text"
        ```
    """

    def _create_agent(response: str) -> Agent:
        model = TestModel(custom_result_text=response)
        return Agent(model, instructions='Test agent')

    return _create_agent
