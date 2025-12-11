# PydanticAI Project Structure Best Practices

> A comprehensive guide to scaffolding production-grade PydanticAI applications based on official Pydantic patterns and community best practices.

**Document Version:** 1.0  
**Last Verified:** December 2025  
**Documentation Sources:** PydanticAI Official Docs, pydantic-settings, Astral uv

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Recommended Project Structure](#recommended-project-structure)
3. [Configuration Management](#configuration-management)
4. [Dependency Injection Patterns](#dependency-injection-patterns)
5. [Testing Infrastructure](#testing-infrastructure)
6. [Observability Setup](#observability-setup)
7. [Package Management with uv](#package-management-with-uv)
8. [CI/CD Configuration](#cicd-configuration)
9. [Complete Example Project](#complete-example-project)
10. [Appendix: References](#appendix-references)

---

## Executive Summary

### Current State (December 2025)

Pydantic and PydanticAI do not provide an official project template or scaffolding tool. However, the Pydantic team maintains several reference repositories that demonstrate production patterns:

| Repository | Purpose | Key Patterns |
|------------|---------|--------------|
| `pydantic/pydantic-ai` | Main framework | uv workspace monorepo |
| `pydantic/logfire-demo` | Full-stack demo | FastAPI + Docker + Observability |
| `pydantic/ai-chat-ui` | Chat frontend | React + Vercel AI Protocol |

This guide synthesizes patterns from these repositories and official documentation to provide actionable scaffolding guidance.

### Key Principles

1. **Type Safety First** — Leverage Pydantic's validation at every layer
2. **Dependency Injection** — Use dataclasses for type-safe service containers
3. **Environment-Driven Config** — pydantic-settings for all configuration
4. **Test Without API Calls** — TestModel and FunctionModel for deterministic tests
5. **Observability Built-In** — Logfire integration from day one

---

## Recommended Project Structure

### Single Application Structure

```
my-pydanticai-app/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Tests, linting, type checking
│       └── deploy.yml              # Deployment (optional)
├── src/
│   └── my_app/
│       ├── __init__.py
│       ├── agents/                 # PydanticAI agents
│       │   ├── __init__.py
│       │   ├── support_agent.py
│       │   └── research_agent.py
│       ├── models/                 # Pydantic data models
│       │   ├── __init__.py
│       │   ├── schemas.py          # Input/output schemas
│       │   └── responses.py        # Structured outputs
│       ├── tools/                  # Agent tools
│       │   ├── __init__.py
│       │   ├── database.py
│       │   └── external_api.py
│       ├── services/               # Business logic
│       │   ├── __init__.py
│       │   └── orchestrator.py
│       ├── config.py               # pydantic-settings configuration
│       ├── dependencies.py         # Dependency containers
│       └── main.py                 # Entry point / FastAPI app
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures + ALLOW_MODEL_REQUESTS
│   ├── test_agents.py
│   └── test_tools.py
├── .env.example                    # Environment template
├── .gitignore
├── .pre-commit-config.yaml         # Code quality hooks
├── .python-version                 # Python version (pyenv/uv)
├── CLAUDE.md                       # AI assistant instructions (optional)
├── Dockerfile                      # Container build (optional)
├── Makefile                        # Development automation
├── README.md
├── pyproject.toml                  # Project configuration
└── uv.lock                         # Dependency lock file
```

### Monorepo Structure (Multiple Packages)

For larger projects with shared components:

```
my-pydanticai-workspace/
├── packages/
│   ├── core/                       # Shared models and utilities
│   │   ├── src/core/
│   │   └── pyproject.toml
│   ├── agents/                     # Agent definitions
│   │   ├── src/agents/
│   │   └── pyproject.toml
│   └── api/                        # FastAPI application
│       ├── src/api/
│       └── pyproject.toml
├── tests/
├── .github/workflows/
├── pyproject.toml                  # Workspace root
└── uv.lock
```

---

## Configuration Management

### pydantic-settings Integration

Use `pydantic-settings` for type-safe, environment-driven configuration:

```python
# src/my_app/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """Configuration for AI agents with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_prefix='AGENT_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )
    
    # Model selection
    primary_model: str = Field(
        default='anthropic:claude-sonnet-4-5',
        description='Primary LLM model identifier'
    )
    fallback_model: str = Field(
        default='openai:gpt-4o',
        description='Fallback model for retries'
    )
    
    # Model settings
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: float = Field(default=60.0, ge=1.0)
    
    # API keys (loaded from environment)
    openai_api_key: str | None = Field(default=None, alias='OPENAI_API_KEY')
    anthropic_api_key: str | None = Field(default=None, alias='ANTHROPIC_API_KEY')


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix='DB_',
        env_file='.env',
    )
    
    host: str = Field(default='localhost')
    port: int = Field(default=5432)
    name: str = Field(default='myapp')
    user: str = Field(default='postgres')
    password: str = Field(default='')
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class AppConfig(BaseSettings):
    """Root application configuration."""
    
    model_config = SettingsConfigDict(env_file='.env')
    
    debug: bool = Field(default=False)
    log_level: str = Field(default='INFO')
    
    # Nested configurations
    agent: AgentConfig = Field(default_factory=AgentConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


# Singleton pattern for configuration
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get application configuration (singleton)."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
```

### Environment File Template

```bash
# .env.example

# Agent Configuration
AGENT_PRIMARY_MODEL=anthropic:claude-sonnet-4-5
AGENT_FALLBACK_MODEL=openai:gpt-4o
AGENT_MAX_TOKENS=4096
AGENT_TEMPERATURE=0.7

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=myapp
DB_USER=postgres
DB_PASSWORD=secret

# Observability
LOGFIRE_TOKEN=...

# Application
DEBUG=false
LOG_LEVEL=INFO
```

---

## Dependency Injection Patterns

### Dataclass-Based Dependencies

PydanticAI uses dataclasses for type-safe dependency injection:

```python
# src/my_app/dependencies.py
from dataclasses import dataclass
from typing import Protocol

import httpx

from my_app.config import AppConfig, get_config


class DatabaseProtocol(Protocol):
    """Protocol for database operations."""
    
    async def get_customer(self, customer_id: int) -> dict | None: ...
    async def save_result(self, result: dict) -> None: ...


@dataclass
class AppDependencies:
    """Container for application dependencies."""
    
    config: AppConfig
    http_client: httpx.AsyncClient
    database: DatabaseProtocol
    
    @classmethod
    async def create(cls) -> "AppDependencies":
        """Factory method for creating dependencies."""
        config = get_config()
        http_client = httpx.AsyncClient(timeout=config.agent.timeout)
        database = await create_database_connection(config.database)
        
        return cls(
            config=config,
            http_client=http_client,
            database=database,
        )
    
    async def close(self) -> None:
        """Cleanup resources."""
        await self.http_client.aclose()


@dataclass
class AgentDependencies:
    """Dependencies specific to agent operations."""
    
    customer_id: int
    database: DatabaseProtocol
    http_client: httpx.AsyncClient
    api_key: str
```

### Using Dependencies in Agents

```python
# src/my_app/agents/support_agent.py
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from my_app.dependencies import AgentDependencies


class SupportOutput(BaseModel):
    """Structured output for support queries."""
    
    advice: str = Field(description='Support advice for the customer')
    risk_level: int = Field(description='Risk assessment (0-10)', ge=0, le=10)
    escalate: bool = Field(description='Whether to escalate to human')


support_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    deps_type=AgentDependencies,
    output_type=SupportOutput,
    instructions='You are a helpful customer support agent.',
)


@support_agent.instructions
async def add_customer_context(ctx: RunContext[AgentDependencies]) -> str:
    """Add customer information to instructions."""
    customer = await ctx.deps.database.get_customer(ctx.deps.customer_id)
    if customer:
        return f"Customer name: {customer['name']}, Account type: {customer['type']}"
    return "Customer information not available."


@support_agent.tool
async def get_account_balance(
    ctx: RunContext[AgentDependencies],
    include_pending: bool = False,
) -> str:
    """Get customer's account balance."""
    # Tool implementation using dependencies
    response = await ctx.deps.http_client.get(
        f"https://api.example.com/balance/{ctx.deps.customer_id}",
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
        params={"include_pending": include_pending},
    )
    response.raise_for_status()
    data = response.json()
    return f"${data['balance']:.2f}"
```

---

## Testing Infrastructure

### Test Configuration

```python
# tests/conftest.py
import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from my_app.agents.support_agent import support_agent
from my_app.dependencies import AgentDependencies


# CRITICAL: Block all real model requests during tests
@pytest.fixture(autouse=True)
def block_real_model_requests():
    """Prevent any real LLM API calls during tests."""
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    yield
    models.ALLOW_MODEL_REQUESTS = original


@pytest.fixture
def test_model():
    """Provide a TestModel instance for mocking."""
    return TestModel()


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    class MockDatabase:
        async def get_customer(self, customer_id: int) -> dict | None:
            if customer_id == 123:
                return {"name": "Test User", "type": "premium"}
            return None
        
        async def save_result(self, result: dict) -> None:
            pass
    
    return MockDatabase()


@pytest.fixture
def agent_deps(mock_database):
    """Create test dependencies."""
    import httpx
    
    return AgentDependencies(
        customer_id=123,
        database=mock_database,
        http_client=httpx.AsyncClient(),
        api_key="test-key",
    )


@pytest.fixture
def override_support_agent(test_model):
    """Override support agent with test model."""
    with support_agent.override(model=test_model):
        yield
```

### Testing with TestModel

```python
# tests/test_agents.py
import pytest
from pydantic_ai.models.test import TestModel

from my_app.agents.support_agent import support_agent, SupportOutput


@pytest.mark.asyncio
async def test_support_agent_basic(agent_deps, override_support_agent):
    """Test basic agent functionality with TestModel."""
    result = await support_agent.run(
        "What is my account balance?",
        deps=agent_deps,
    )
    
    # TestModel returns structured output with defaults
    assert isinstance(result.output, SupportOutput)
    assert result.usage().requests >= 1


@pytest.mark.asyncio
async def test_support_agent_custom_output(agent_deps):
    """Test with custom output arguments."""
    test_model = TestModel(
        custom_output_args={
            "advice": "Your balance is $500.00",
            "risk_level": 2,
            "escalate": False,
        }
    )
    
    with support_agent.override(model=test_model):
        result = await support_agent.run(
            "Check my balance",
            deps=agent_deps,
        )
    
    assert result.output.advice == "Your balance is $500.00"
    assert result.output.risk_level == 2
    assert result.output.escalate is False
```

### Testing with FunctionModel

```python
# tests/test_agents_advanced.py
import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from my_app.agents.support_agent import support_agent


@pytest.mark.asyncio
async def test_agent_tool_calls(agent_deps):
    """Test agent behavior with simulated tool calls."""
    
    def simulate_tool_call(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        """Simulate model that calls tools then responds."""
        # Check if we've received tool results
        last_message = messages[-1] if messages else None
        has_tool_results = any(
            hasattr(part, 'part_kind') and part.part_kind == 'tool-return'
            for part in getattr(last_message, 'parts', [])
        )
        
        if has_tool_results:
            # Return final structured response
            return ModelResponse(parts=[
                TextPart(content='{"advice": "Based on your balance...", "risk_level": 1, "escalate": false}')
            ])
        
        # First call - invoke the balance tool
        return ModelResponse(parts=[
            ToolCallPart(
                tool_name='get_account_balance',
                args={'include_pending': True},
                tool_call_id='call-1',
            )
        ])
    
    with support_agent.override(model=FunctionModel(simulate_tool_call)):
        result = await support_agent.run(
            "What's my current balance?",
            deps=agent_deps,
        )
    
    # Verify tool was called
    assert result.usage().tool_calls >= 1
```

---

## Observability Setup

### Logfire Integration

```python
# src/my_app/observability.py
import logfire

from my_app.config import get_config


def setup_observability() -> None:
    """Configure Logfire observability."""
    config = get_config()
    
    # Configure Logfire
    logfire.configure(
        service_name='my-pydanticai-app',
        environment='development' if config.debug else 'production',
    )
    
    # Instrument PydanticAI agents
    logfire.instrument_pydantic_ai()
    
    # Instrument HTTP clients
    logfire.instrument_httpx(capture_all=True)
    
    # Instrument database (if using asyncpg)
    # logfire.instrument_asyncpg()


# Alternative: OpenTelemetry without Logfire
def setup_otel_observability() -> None:
    """Configure raw OpenTelemetry tracing."""
    import os
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import set_tracer_provider
    from pydantic_ai import Agent
    
    os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318'
    
    exporter = OTLPSpanExporter()
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(span_processor)
    set_tracer_provider(tracer_provider)
    
    # Instrument all agents
    Agent.instrument_all()
```

### Application Entry Point with Observability

```python
# src/my_app/main.py
import asyncio

from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager

from my_app.config import get_config
from my_app.dependencies import AppDependencies
from my_app.observability import setup_observability
from my_app.agents.support_agent import support_agent, AgentDependencies


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_observability()
    app.state.deps = await AppDependencies.create()
    
    yield
    
    # Shutdown
    await app.state.deps.close()


app = FastAPI(
    title="My PydanticAI App",
    lifespan=lifespan,
)


@app.post("/support")
async def handle_support_query(
    query: str,
    customer_id: int,
):
    """Handle customer support queries."""
    config = get_config()
    
    agent_deps = AgentDependencies(
        customer_id=customer_id,
        database=app.state.deps.database,
        http_client=app.state.deps.http_client,
        api_key=config.agent.openai_api_key or "",
    )
    
    result = await support_agent.run(query, deps=agent_deps)
    
    return {
        "response": result.output.model_dump(),
        "usage": {
            "requests": result.usage().requests,
            "input_tokens": result.usage().input_tokens,
            "output_tokens": result.usage().output_tokens,
        }
    }
```

---

## Package Management with uv

### pyproject.toml Configuration

```toml
[project]
name = "my-pydanticai-app"
version = "0.1.0"
description = "Production PydanticAI application"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "you@example.com" }
]

dependencies = [
    "pydantic-ai>=1.0.0",
    "pydantic-settings>=2.5.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "httpx>=0.27.0",
    "logfire>=3.6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "pytest-cov>=5.0.0",
    "dirty-equals>=0.9.0",
    "inline-snapshot>=0.19.0",
]
lint = [
    "ruff>=0.8.0",
    "pyright>=1.1.390",
    "pre-commit>=4.0.0",
]
docs = [
    "mkdocs>=1.6.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings-python>=1.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/my_app"]

# Ruff configuration
[tool.ruff]
line-length = 120
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "I",      # isort
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "SIM",    # flake8-simplify
    "ASYNC",  # flake8-async
]

[tool.ruff.lint.isort]
known-first-party = ["my_app"]

# Pyright configuration
[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "strict"
include = ["src", "tests"]

# Pytest configuration
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = [
    "--strict-markers",
    "-ra",
]

# Coverage configuration
[tool.coverage.run]
source = ["src/my_app"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

### Makefile for Development Automation

```makefile
.PHONY: install dev lint test format clean

# Install production dependencies
install:
	uv sync --locked

# Install with development dependencies
dev:
	uv sync --locked --all-extras --dev

# Run linting and type checking
lint:
	uv run ruff check src tests
	uv run pyright src tests

# Run tests with coverage
test:
	uv run pytest --cov --cov-report=term-missing

# Format code
format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

# Run the application
run:
	uv run uvicorn my_app.main:app --reload

# Clean build artifacts
clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name "__pycache__" -exec rm -rf {} +
```

---

## CI/CD Configuration

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]

    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --locked --all-extras --dev

      - name: Run linting
        run: |
          uv run ruff check src tests
          uv run ruff format --check src tests

      - name: Run type checking
        run: uv run pyright src tests

      - name: Run tests
        run: uv run pytest --cov --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: uv run pyright
        language: system
        types: [python]
        pass_filenames: false
```

---

## Complete Example Project

### Project Summary

A complete, minimal PydanticAI project implementing the patterns described:

```
my-pydanticai-app/
├── .github/workflows/ci.yml
├── src/my_app/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── support_agent.py
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   └── observability.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_agents.py
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── Makefile
├── README.md
└── pyproject.toml
```

### Quick Start Commands

```bash
# Clone and setup
git clone <your-repo>
cd my-pydanticai-app

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync --dev

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run tests
uv run pytest

# Start development server
uv run uvicorn my_app.main:app --reload

# Run linting
uv run ruff check src tests
uv run pyright src tests
```

---

## Appendix: References

### A.1 PydanticAI Official Documentation

| Topic | Source | Doc ID |
|-------|--------|--------|
| Unit Testing | PydanticAI Docs | `PydanticAI_0116` |
| Dependencies | PydanticAI Docs | `PydanticAI_0011` |
| TestModel API | PydanticAI Docs | `PydanticAI_0074` |
| Logfire Integration | PydanticAI Docs | `PydanticAI_0115` |
| Installation | PydanticAI Docs | `PydanticAI_0004` |

**Key Documentation URLs:**
- https://ai.pydantic.dev/testing/
- https://ai.pydantic.dev/dependencies/
- https://ai.pydantic.dev/logfire/

### A.2 pydantic-settings Documentation

| Topic | Library ID | Snippet Count |
|-------|------------|---------------|
| BaseSettings Configuration | `/pydantic/pydantic-settings` | 67 |
| Environment Variables | `/pydantic/pydantic-settings` | 67 |
| Nested Configuration | `/pydantic/pydantic-settings` | 67 |

**Key Patterns Verified:**
- `SettingsConfigDict` with `env_prefix`, `env_file`
- `Field(validation_alias=...)` for custom env var names
- `env_nested_delimiter` for nested models

### A.3 Astral uv Documentation

| Topic | Library ID | Benchmark Score |
|-------|------------|-----------------|
| Project Configuration | `/astral-sh/uv` | 87.2 |
| Workspace Management | `/astral-sh/uv` | 87.2 |
| Dependency Groups | `/astral-sh/uv` | 87.2 |

**Key Patterns Verified:**
- `[dependency-groups]` for dev/lint/test dependencies
- `[tool.uv.workspace]` for monorepo setups
- `uv sync --locked --dev` for reproducible installs

### A.4 GitHub Repository References

| Repository | URL | Last Verified |
|------------|-----|---------------|
| pydantic-ai | https://github.com/pydantic/pydantic-ai | 2025-12-10 |
| logfire-demo | https://github.com/pydantic/logfire-demo | 2025-12-10 |
| ai-chat-ui | https://github.com/pydantic/ai-chat-ui | 2025-12-10 |
| pydantic-settings | https://github.com/pydantic/pydantic-settings | 2025-12-10 |

### A.5 Verification Tools Used

| Tool | Purpose |
|------|---------|
| `qdrant-docs (FastMCP)` | PydanticAI documentation retrieval |
| `Context7` | pydantic-settings, uv documentation |
| `web_search` | GitHub repository verification |
| `mcp-server-time` | Timestamp verification |

---

**Document Generated:** 2025-12-10 19:30 PST  
**Verification Status:** All patterns verified against official documentation sources
