"""Tests for PydanticAI agents.

These tests demonstrate how to test agents using TestModel without making API calls.
"""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


@pytest.mark.asyncio
async def test_basic_agent(basic_test_agent: Agent):
    """Test that a basic agent can run successfully."""
    result = await basic_test_agent.run('Hello, how are you?')
    assert result.output is not None
    assert isinstance(result.output, str)


@pytest.mark.asyncio
async def test_agent_with_custom_response(agent_with_custom_response):
    """Test agent with a predefined response."""
    expected_response = 'This is a test response'
    agent = agent_with_custom_response(expected_response)

    result = await agent.run('Test query')
    assert result.output == expected_response


@pytest.mark.asyncio
async def test_agent_with_tools():
    """Test an agent with tools using TestModel."""
    test_model = TestModel()
    agent = Agent(test_model, instructions='You are a calculator assistant.')

    @agent.tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    # Note: TestModel doesn't actually call tools, but we can verify the agent structure
    assert len(agent._function_tools) == 1
    assert 'add_numbers' in [tool.name for tool in agent._function_tools.values()]


@pytest.mark.asyncio
async def test_test_model_usage():
    """Test direct usage of TestModel."""
    model = TestModel(custom_result_text='Custom output')

    # Create a simple agent
    agent = Agent(model, instructions='Test instructions')
    result = await agent.run('Any query')

    assert result.output == 'Custom output'


@pytest.mark.asyncio
async def test_config_import():
    """Test that config module can be imported and used."""
    from pydantic_learning.config import settings

    # Verify settings exist and have expected attributes
    assert hasattr(settings, 'default_model')
    assert hasattr(settings, 'answerer_model')
    assert hasattr(settings, 'questioner_model')
    assert hasattr(settings, 'search_model')
    assert hasattr(settings, 'analysis_model')

    # Verify default values are set to gpt-5-nano
    assert settings.default_model == 'openai:gpt-5-nano'
    assert settings.answerer_model == 'openai:gpt-5-nano'
    assert settings.questioner_model == 'openai:gpt-5-nano'
    assert settings.search_model == 'openai:gpt-5-nano'
    assert settings.analysis_model == 'openai:gpt-5-nano'

    # Verify model settings
    assert settings.temperature == 0.7
    assert settings.max_tokens == 4096
    assert settings.timeout == 60.0


@pytest.mark.asyncio
async def test_agent_with_config():
    """Test creating an agent using config settings with TestModel override."""
    from pydantic_learning.config import settings

    # In tests, we override the model from config with TestModel
    test_model = TestModel()
    agent = Agent(
        test_model,  # Use TestModel instead of settings.default_model
        instructions='You are a helpful assistant.',
    )

    result = await agent.run('Test query')
    assert result.output is not None
