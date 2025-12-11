"""Pytest configuration and fixtures for PydanticAI Learning Hub tests.

This module provides fixtures for testing agents without making actual API calls.
The ALLOW_MODEL_REQUESTS environment variable controls whether real API calls are allowed.

Best Practices (per PydanticAI docs as of December 2025):
- Use `TestModel` for unit tests without API calls
- Use `agent.override(model=TestModel())` for testing existing agents
- Set `models.ALLOW_MODEL_REQUESTS = False` to block real API calls globally
- Use `custom_output_text` parameter for predefined responses
- Use `@agent.tool_plain` for tools without RunContext access
- Use `@agent.tool` for tools that need RunContext[DepsType]
"""

import os
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent, models
from pydantic_ai.models.test import TestModel

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Block all real model requests during tests (PydanticAI best practice)
models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def allow_model_requests():
    """Fixture that enables real model requests for integration tests.

    Use this fixture when you need to test against actual LLM APIs.
    By default, all model requests are blocked (ALLOW_MODEL_REQUESTS=false).

    Usage:
        @pytest.mark.asyncio
        async def test_with_real_api(allow_model_requests):
            # Real API calls are now allowed for this test
            ...

    Yields:
        bool: True if real requests were enabled, False otherwise
    """
    should_allow = os.getenv('ALLOW_MODEL_REQUESTS', 'false').lower() == 'true'
    if should_allow:
        models.ALLOW_MODEL_REQUESTS = True
        yield True
        models.ALLOW_MODEL_REQUESTS = False
    else:
        yield False


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
def agent_with_custom_response() -> callable:
    """Factory fixture for creating agents with custom test responses.

    This follows PydanticAI best practices by using TestModel with custom_output_text.

    Returns:
        callable: A function that creates an agent with a predefined response

    Example:
        ```python
        @pytest.mark.asyncio
        async def test_something(agent_with_custom_response):
            agent = agent_with_custom_response("Custom response text")
            result = await agent.run("Test query")
            assert result.output == "Custom response text"
        ```
    """

    def _create_agent(response: str) -> Agent:
        model = TestModel(custom_output_text=response)
        return Agent(model, instructions='Test agent')

    return _create_agent


@pytest.fixture
def override_with_test_model():
    """Factory fixture to override any agent's model with TestModel.

    This is the recommended pattern from PydanticAI docs for testing
    existing agents without modifying their source code.

    Example:
        ```python
        from my_app import my_agent

        @pytest.mark.asyncio
        async def test_my_agent(override_with_test_model):
            with override_with_test_model(my_agent):
                result = await my_agent.run("test query")
                assert result.output == 'success (no tool calls)'
        ```
    """
    from contextlib import contextmanager

    @contextmanager
    def _override(agent: Agent, custom_output_text: str | None = None):
        test_model = TestModel(custom_output_text=custom_output_text)
        with agent.override(model=test_model):
            yield test_model

    return _override
