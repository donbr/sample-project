# PydanticAI Best Practices and Patterns

> A comprehensive guide to configuration, testing, and production patterns for PydanticAI agent development.

**Last Updated:** December 2025  
**PydanticAI Version:** 1.x  
**Documentation Source:** Official PydanticAI Documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Model Configuration](#model-configuration)
   - [Environment Variable Patterns](#environment-variable-patterns)
   - [Runtime Model Override](#runtime-model-override)
   - [Pydantic Settings Integration](#pydantic-settings-integration)
3. [ModelSettings Reference](#modelsettings-reference)
   - [Cross-Provider Settings](#cross-provider-settings)
   - [Provider-Specific Settings](#provider-specific-settings)
   - [Provider Support Matrix](#provider-support-matrix)
4. [Testing Patterns](#testing-patterns)
   - [TestModel - Simple Mocking](#testmodel---simple-mocking)
   - [FunctionModel - Advanced Mocking](#functionmodel---advanced-mocking)
   - [Agent.override - Dependency Injection](#agentoverride---dependency-injection)
   - [ALLOW_MODEL_REQUESTS Safety Guard](#allow_model_requests-safety-guard)
5. [Multi-Agent Patterns](#multi-agent-patterns)
   - [Agent Delegation](#agent-delegation)
   - [Programmatic Hand-off](#programmatic-hand-off)
6. [Production Patterns](#production-patterns)
   - [Agent Factory Pattern](#agent-factory-pattern)
   - [Complete Configuration Example](#complete-configuration-example)
7. [Observability](#observability)
8. [Quick Reference](#quick-reference)

---

## Overview

PydanticAI is a Python agent framework designed to make building production-grade GenAI applications straightforward. This document captures best practices for:

- **Configuration management** - Environment variables, settings, and model selection
- **Testing strategies** - Unit testing without real API calls
- **Production patterns** - Factory patterns, dependency injection, and observability

---

## Model Configuration

PydanticAI supports multiple approaches for configuring LLM models, from simple environment variables to comprehensive settings classes.

### Environment Variable Patterns

#### PYDANTIC_AI_MODEL Override

The simplest approach for quick model switching:

```bash
# Override model at runtime
PYDANTIC_AI_MODEL=gemini-2.5-pro python my_app.py

# With different providers
PYDANTIC_AI_MODEL=anthropic:claude-sonnet-4-5 python my_app.py
PYDANTIC_AI_MODEL=openai:gpt-4o python my_app.py
```

#### Reading from Environment

```python
import os
from pydantic_ai import Agent

# Read model from environment with fallback
model = os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-4o')

agent = Agent(
    model,
    instructions='Your instructions here',
)
```

#### Provider-Specific API Keys

Each provider uses standard environment variables:

```bash
# Provider API keys
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
export GOOGLE_API_KEY='your-google-key'
export GROQ_API_KEY='your-groq-key'
export MISTRAL_API_KEY='your-mistral-key'
export CO_API_KEY='your-cohere-key'
```

### Runtime Model Override

Override the model at runtime without modifying agent construction:

```python
from pydantic_ai import Agent

# Define agent with a default model
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Analyze queries and design research plans.',
)

# Override at runtime
result = await agent.run(
    'Your query',
    model='openai:gpt-4o',  # Runtime override
)

# Override with model settings
result = await agent.run(
    'Your query',
    model='anthropic:claude-opus-4-5',
    model_settings={'temperature': 0.2, 'max_tokens': 8192},
)
```

### Pydantic Settings Integration

For production applications, use `pydantic-settings` for comprehensive configuration:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


class AgentConfig(BaseSettings):
    """Configuration for AI agents with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_prefix='AGENT_',
        env_file='.env',
        env_file_encoding='utf-8',
    )
    
    # Model configurations
    plan_model: str = Field(default='anthropic:claude-sonnet-4-5')
    search_model: str = Field(default='google-vertex:gemini-2.5-flash')
    analysis_model: str = Field(default='anthropic:claude-sonnet-4-5')
    
    # Model settings (cross-provider)
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: float = Field(default=30.0)
    
    def to_model_settings(self) -> ModelSettings:
        """Convert to PydanticAI ModelSettings."""
        return ModelSettings(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )


# Usage
config = AgentConfig()

agent = Agent(
    config.plan_model,
    instructions='Your instructions here',
    model_settings=config.to_model_settings(),
)
```

**Environment variables (`.env` file):**

```bash
AGENT_PLAN_MODEL=anthropic:claude-opus-4-5
AGENT_SEARCH_MODEL=openai:gpt-4o
AGENT_MAX_TOKENS=8192
AGENT_TEMPERATURE=0.5
```

---

## ModelSettings Reference

### Cross-Provider Settings

The `ModelSettings` TypedDict contains settings that work across multiple providers:

```python
from pydantic_ai.settings import ModelSettings

settings = ModelSettings(
    max_tokens=4096,        # Maximum tokens to generate
    temperature=0.7,        # Randomness (0.0 = deterministic)
    top_p=0.9,             # Nucleus sampling threshold
    timeout=30.0,          # Request timeout in seconds
    parallel_tool_calls=True,  # Allow parallel tool execution
    seed=42,               # Random seed for reproducibility
)
```

#### Settings Descriptions

| Setting | Type | Description | Default Guidance |
|---------|------|-------------|------------------|
| `max_tokens` | `int` | Maximum tokens to generate before stopping | 4096 for most use cases |
| `temperature` | `float` | Randomness: 0.0 = deterministic, 2.0 = very creative | 0.7 balanced, 0.0-0.3 for analytical |
| `top_p` | `float` | Nucleus sampling: cumulative probability threshold | 0.9 typical |
| `timeout` | `float` | Request timeout in seconds | 30.0 for standard, 120.0 for complex |
| `parallel_tool_calls` | `bool` | Allow model to call multiple tools simultaneously | `True` when tools are independent |
| `seed` | `int` | Random seed for reproducibility | Use for testing only |

### Provider-Specific Settings

Each provider extends `ModelSettings` with prefixed settings:

#### Anthropic Settings

```python
from pydantic_ai.models.anthropic import AnthropicModelSettings

settings = AnthropicModelSettings(
    # Base settings
    max_tokens=4096,
    temperature=0.7,
    
    # Anthropic-specific (prefixed with `anthropic_`)
    anthropic_metadata={'user_id': 'user-123'},
    anthropic_thinking={'enabled': True, 'budget_tokens': 10000},
)
```

#### OpenAI Settings

```python
from pydantic_ai.models.openai import OpenAIModelSettings

settings = OpenAIModelSettings(
    # Base settings
    max_tokens=4096,
    temperature=0.7,
    
    # OpenAI-specific (prefixed with `openai_`)
    openai_reasoning_effort='medium',  # For o1/o3 models: 'low', 'medium', 'high'
    openai_truncation='auto',
    openai_user='user-123',
)
```

#### Google/Gemini Settings

```python
from pydantic_ai.models.google import GoogleModelSettings

settings = GoogleModelSettings(
    # Base settings
    max_tokens=4096,
    temperature=0.7,
    
    # Google-specific (prefixed with `google_`)
    google_safety_settings=[
        {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'}
    ],
)
```

#### Merging Multi-Provider Settings

Settings use prefixes, so you can create unified settings for multi-provider scenarios:

```python
# Unified settings - each provider ignores irrelevant prefixed settings
unified_settings: ModelSettings = {
    # Base settings (work everywhere)
    'max_tokens': 4096,
    'temperature': 0.7,
    
    # Anthropic-specific (ignored by OpenAI, Google, etc.)
    'anthropic_metadata': {'user_id': 'user-123'},
    
    # OpenAI-specific (ignored by Anthropic, Google, etc.)
    'openai_reasoning_effort': 'medium',
}

# Use with any agent
anthropic_agent = Agent('anthropic:claude-sonnet-4-5', model_settings=unified_settings)
openai_agent = Agent('openai:gpt-4o', model_settings=unified_settings)
```

### Provider Support Matrix

| Setting | Anthropic | OpenAI | Google | Groq | Cohere | Mistral | Bedrock |
|---------|:---------:|:------:|:------:|:----:|:------:|:-------:|:-------:|
| `max_tokens` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `temperature` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `top_p` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `timeout` | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| `parallel_tool_calls` | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| `seed` | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |

---

## Testing Patterns

PydanticAI provides comprehensive testing support to avoid real API calls during tests.

### TestModel - Simple Mocking

`TestModel` provides predictable, static responses:

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-4o')

# Basic usage - returns 'success' by default
result = agent.run_sync('What is the capital of France?', model=TestModel())
print(result.output)  # 'success'
```

#### TestModel with Structured Output

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


class CityInfo(BaseModel):
    name: str
    country: str


agent = Agent('openai:gpt-4o', output_type=CityInfo)

# Provide custom output arguments
result = agent.run_sync(
    'Tell me about Paris',
    model=TestModel(custom_output_args={'name': 'Paris', 'country': 'France'})
)
print(result.output)  # CityInfo(name='Paris', country='France')
```

#### TestModel with Custom Text

```python
result = agent.run_sync(
    'Hello',
    model=TestModel(custom_output_text='Custom response text')
)
print(result.output)  # 'Custom response text'
```

### FunctionModel - Advanced Mocking

`FunctionModel` gives complete control over model behavior:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel, AgentInfo

agent = Agent('openai:gpt-4o')


async def my_model_function(
    messages: list[ModelMessage], 
    info: AgentInfo
) -> ModelResponse:
    """Custom model implementation for testing."""
    # Access conversation history
    user_content = str(messages[0]) if messages else 'no message'
    
    # Access agent metadata
    has_tools = len(info.function_tools) > 0
    
    # Return custom response
    return ModelResponse(parts=[
        TextPart(content=f'Processed: {user_content}, tools: {has_tools}')
    ])


result = agent.run_sync('Test query', model=FunctionModel(my_model_function))
```

#### FunctionModel with Tool Calls

```python
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
)
from pydantic_ai.models.function import FunctionModel, AgentInfo

agent = Agent('openai:gpt-4o')


@agent.tool_plain
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"


async def model_with_tool_calls(
    messages: list[ModelMessage], 
    info: AgentInfo
) -> ModelResponse:
    """Simulate model that calls tools."""
    # Check if we've received tool results
    last_message = messages[-1] if messages else None
    has_tool_results = last_message and any(
        isinstance(part, ToolReturnPart) 
        for part in getattr(last_message, 'parts', [])
    )
    
    if has_tool_results:
        # Tool results received - return final response
        return ModelResponse(parts=[TextPart(content='Weather check complete!')])
    
    # No tool results yet - call the weather tool
    return ModelResponse(parts=[
        ToolCallPart(
            tool_name='get_weather',
            args={'city': 'London'},
            tool_call_id='test-call-1'
        )
    ])


result = agent.run_sync('Weather?', model=FunctionModel(model_with_tool_calls))
print(result.output)  # 'Weather check complete!'
```

#### AgentInfo Structure

```python
@dataclass
class AgentInfo:
    function_tools: list[ToolDefinition]  # Available function tools
    output_tools: list[ToolDefinition]    # Output schema tools
    allow_text_output: bool               # Whether plain text is allowed
    model_settings: ModelSettings | None  # Current model settings
```

### Agent.override - Dependency Injection

`Agent.override` is a context manager for replacing model, deps, or toolsets in tests:

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# Production agent
agent = Agent('anthropic:claude-sonnet-4-5', deps_type=str)


def application_function(user_query: str) -> str:
    """Application code that uses the agent."""
    result = agent.run_sync(user_query, deps='production-api-key')
    return result.output


# Test with override
def test_application_function():
    with agent.override(model=TestModel(), deps='test-api-key'):
        result = application_function('Test query')
        assert result == 'success'
```

#### Override Multiple Agents

```python
from contextlib import ExitStack
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

plan_agent = Agent('anthropic:claude-sonnet-4-5', name='plan_agent')
search_agent = Agent('google-vertex:gemini-2.5-flash', name='search_agent')


def test_multi_agent_workflow():
    with ExitStack() as stack:
        stack.enter_context(plan_agent.override(model=TestModel()))
        stack.enter_context(search_agent.override(model=TestModel()))
        
        # Test code here
        result = plan_agent.run_sync('Create a plan')
        assert result.output == 'success'
```

### ALLOW_MODEL_REQUESTS Safety Guard

Block all real model requests globally:

```python
# conftest.py
import pytest
from pydantic_ai.models import ALLOW_MODEL_REQUESTS


@pytest.fixture(autouse=True)
def block_real_model_requests():
    """Prevent any real model API calls during tests."""
    with ALLOW_MODEL_REQUESTS.set(False):
        yield


@pytest.fixture
def allow_model_requests():
    """Allow real model requests for integration tests."""
    with ALLOW_MODEL_REQUESTS.set(True):
        yield
```

**Usage:**

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


def test_with_test_model():
    """Works - TestModel doesn't make real requests."""
    agent = Agent('openai:gpt-4o')
    result = agent.run_sync('Hello', model=TestModel())
    assert result.output == 'success'


def test_accidentally_using_real_model():
    """Fails - ALLOW_MODEL_REQUESTS=False blocks real calls."""
    agent = Agent('openai:gpt-4o')
    with pytest.raises(RuntimeError, match='Model requests are not allowed'):
        agent.run_sync('Hello')  # No test model provided!


@pytest.mark.usefixtures('allow_model_requests')
def test_integration_with_real_model():
    """Integration test that actually calls the API."""
    agent = Agent('openai:gpt-4o')
    result = agent.run_sync('Say hello')
    assert 'hello' in result.output.lower()
```

---

## Multi-Agent Patterns

### Agent Delegation

Agents can delegate work to other agents via tools:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import AbstractAgent

# Delegate agent
search_agent = Agent(
    'google-vertex:gemini-2.5-flash',
    instructions='Perform web searches and return results.',
)

# Main agent with delegation capability
analysis_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    deps_type=AbstractAgent,
    instructions='Analyze data. Use extra_search tool if needed.',
)


@analysis_agent.tool
async def extra_search(ctx: RunContext[AbstractAgent], query: str) -> str:
    """Perform an additional search via the delegate agent."""
    result = await ctx.deps.run(query)
    return result.output


# Usage - pass search_agent as dependency
result = await analysis_agent.run(
    'Analyze hedge funds in London',
    deps=search_agent,
)
```

### Programmatic Hand-off

Application code orchestrates multiple agents:

```python
import asyncio
from pydantic_ai import Agent, format_as_xml

plan_agent = Agent('anthropic:claude-sonnet-4-5', output_type=ResearchPlan)
search_agent = Agent('google-vertex:gemini-2.5-flash')
analysis_agent = Agent('anthropic:claude-sonnet-4-5')


async def deep_research(query: str) -> str:
    # Phase 1: Planning
    plan_result = await plan_agent.run(query)
    plan = plan_result.output
    
    # Phase 2: Parallel execution
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(search_agent.run(step.search_terms))
            for step in plan.web_search_steps
        ]
    search_results = [task.result().output for task in tasks]
    
    # Phase 3: Analysis
    analysis_result = await analysis_agent.run(
        format_as_xml({
            'query': query,
            'search_results': search_results,
            'instructions': plan.analysis_instructions,
        })
    )
    return analysis_result.output
```

---

## Production Patterns

### Agent Factory Pattern

Centralize agent creation with a factory:

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.settings import ModelSettings


@dataclass
class AgentFactory:
    """Factory for creating configured agents."""
    config: AgentConfig
    
    def _model_settings(self) -> ModelSettings:
        return ModelSettings(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )
    
    def create_plan_agent(self) -> Agent[None, DeepResearchPlan]:
        return Agent(
            self.config.plan_model,
            instructions='Analyze queries and design research plans.',
            output_type=DeepResearchPlan,
            model_settings=self._model_settings(),
            name='plan_agent',
        )
    
    def create_search_agent(self) -> Agent[None, str]:
        return Agent(
            self.config.search_model,
            instructions='Perform web searches and return detailed reports.',
            builtin_tools=[WebSearchTool()],
            model_settings=self._model_settings(),
            name='search_agent',
        )
    
    def create_analysis_agent(self) -> Agent[AbstractAgent, str]:
        agent = Agent(
            self.config.analysis_model,
            deps_type=AbstractAgent,
            instructions='Analyze research and generate reports.',
            model_settings=self._model_settings(),
            name='analysis_agent',
        )
        
        @agent.tool
        async def extra_search(ctx: RunContext[AbstractAgent], query: str) -> str:
            result = await ctx.deps.run(query)
            return result.output
        
        return agent
```

### Complete Configuration Example

Full production setup with configuration, factory, and observability:

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated

import logfire
from annotated_types import MaxLen
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai import Agent, RunContext, WebSearchTool, format_as_xml
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.settings import ModelSettings


# =============================================================================
# Configuration
# =============================================================================

class DeepResearchConfig(BaseSettings):
    """Environment-driven configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix='DEEP_RESEARCH_',
        env_file='.env',
        extra='ignore',
    )
    
    # Model selection
    plan_model: str = Field(default='anthropic:claude-sonnet-4-5')
    search_model: str = Field(default='google-vertex:gemini-2.5-flash')
    analysis_model: str = Field(default='anthropic:claude-sonnet-4-5')
    
    # Model settings
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    timeout: float = Field(default=60.0)
    
    # Observability
    logfire_enabled: bool = Field(default=True)


# =============================================================================
# Data Models
# =============================================================================

class WebSearchStep(BaseModel):
    """A step that performs a web search."""
    search_terms: str


class DeepResearchPlan(BaseModel, **ConfigDict(use_attribute_docstrings=True)):
    """A structured plan for deep research."""
    
    executive_summary: str
    """A summary of the research plan."""
    
    web_search_steps: Annotated[list[WebSearchStep], MaxLen(5)]
    """Web search steps to perform."""
    
    analysis_instructions: str
    """Instructions for the analysis phase."""


# =============================================================================
# Agent Factory
# =============================================================================

@dataclass
class AgentFactory:
    config: DeepResearchConfig
    
    def _model_settings(self) -> ModelSettings:
        return ModelSettings(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )
    
    def create_plan_agent(self) -> Agent[None, DeepResearchPlan]:
        return Agent(
            self.config.plan_model,
            instructions='Analyze queries and design research plans.',
            output_type=DeepResearchPlan,
            model_settings=self._model_settings(),
            name='plan_agent',
        )
    
    def create_search_agent(self) -> Agent[None, str]:
        return Agent(
            self.config.search_model,
            instructions='Perform web searches and return detailed reports.',
            builtin_tools=[WebSearchTool()],
            model_settings=self._model_settings(),
            name='search_agent',
        )
    
    def create_analysis_agent(self) -> Agent[AbstractAgent, str]:
        agent = Agent(
            self.config.analysis_model,
            deps_type=AbstractAgent,
            instructions='Analyze research and generate comprehensive reports.',
            model_settings=self._model_settings(),
            name='analysis_agent',
        )
        
        @agent.tool
        async def extra_search(ctx: RunContext[AbstractAgent], query: str) -> str:
            result = await ctx.deps.run(query)
            return result.output
        
        return agent


# =============================================================================
# Main Application
# =============================================================================

def setup_observability(config: DeepResearchConfig) -> None:
    if config.logfire_enabled:
        logfire.configure()
        logfire.instrument_pydantic_ai()


@logfire.instrument
async def deep_research(query: str, config: DeepResearchConfig | None = None) -> str:
    config = config or DeepResearchConfig()
    setup_observability(config)
    
    factory = AgentFactory(config)
    plan_agent = factory.create_plan_agent()
    search_agent = factory.create_search_agent()
    analysis_agent = factory.create_analysis_agent()
    
    # Phase 1: Planning
    result = await plan_agent.run(query)
    plan = result.output
    
    # Phase 2: Parallel searches
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(search_agent.run(step.search_terms))
            for step in plan.web_search_steps
        ]
    search_results = [task.result().output for task in tasks]
    
    # Phase 3: Analysis
    analysis_result = await analysis_agent.run(
        format_as_xml({
            'query': query,
            'search_results': search_results,
            'instructions': plan.analysis_instructions,
        }),
        deps=search_agent,
    )
    return analysis_result.output


if __name__ == '__main__':
    result = asyncio.run(deep_research('Find hedge funds that use Python in London'))
    print(result)
```

---

## Observability

### Logfire Integration

```python
import logfire
from pydantic_ai import Agent

# Configure Logfire
logfire.configure()
logfire.instrument_pydantic_ai()

# Custom spans
@logfire.instrument
async def my_workflow(query: str) -> str:
    agent = Agent('anthropic:claude-sonnet-4-5')
    result = await agent.run(query)
    return result.output
```

### Key Observability Features

- **Automatic tracing** of all agent runs
- **Tool call tracking** with inputs and outputs
- **Token usage** and latency metrics
- **Error tracking** with full context
- **Custom spans** via `@logfire.instrument`

---

## Quick Reference

### Model Name Format

```
provider:model-name

# Examples
anthropic:claude-sonnet-4-5
openai:gpt-4o
google-vertex:gemini-2.5-flash
groq:llama-3.3-70b-versatile
```

### Essential Imports

```python
# Core
from pydantic_ai import Agent, RunContext, format_as_xml

# Models
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.models import ALLOW_MODEL_REQUESTS

# Settings
from pydantic_ai.settings import ModelSettings

# Messages (for FunctionModel)
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

# Built-in tools
from pydantic_ai import WebSearchTool

# Agent typing
from pydantic_ai.agent import AbstractAgent
```

### Common Patterns Cheat Sheet

| Pattern | When to Use |
|---------|-------------|
| `TestModel()` | Simple tests with predictable outputs |
| `FunctionModel(fn)` | Complex test scenarios, tool call simulation |
| `agent.override(model=...)` | Testing application code that uses agents |
| `ALLOW_MODEL_REQUESTS.set(False)` | Blocking real API calls in test suite |
| `agent.run(model=...)` | Runtime model switching |
| `AgentFactory` | Production multi-agent systems |
| `pydantic-settings` | Environment-based configuration |

---

## Further Reading

- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [Pydantic Logfire](https://pydantic.dev/logfire)
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

---

*This document was generated from verified PydanticAI documentation sources.*
