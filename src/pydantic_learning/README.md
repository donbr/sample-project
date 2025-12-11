# PydanticAI Learning Modules

A structured, progressive learning path for PydanticAI concepts, from basic agent setup to advanced durable execution patterns.

## Overview

This package contains numbered learning modules that teach PydanticAI concepts incrementally. Each module is self-contained, runnable, and builds upon previous lessons.

## Learning Progression

```
01_basic_agent.py          → Minimal "Hello World" agent
        ↓
02_structured_output.py    → output_type with Pydantic models
        ↓
03_agent_with_tools.py     → @agent.tool decorator
        ↓
04_dependency_injection.py → deps_type, RunContext
        ↓
05_multi_agent_game.py     → Agent-to-agent communication
        ↓
06_multi_agent_research.py → Parallel execution, TaskGroup
        ↓
07_durable_dbos.py         → DBOSAgent wrapper
        ↓
08_durable_temporal.py     → TemporalAgent wrapper
        ↓
09_evaluations.py          → pydantic_evals testing
```

## Module Details

### 01 - Basic Agent
**Concepts**: Agent creation, simple prompts, basic output
**Prerequisites**: None
**Run**: `uv run python -m pydantic_learning.agents.01_basic_agent`

### 02 - Structured Output
**Concepts**: Pydantic models as `output_type`, typed responses
**Prerequisites**: Lesson 01
**Run**: `uv run python -m pydantic_learning.agents.02_structured_output`

### 03 - Agent with Tools
**Concepts**: `@agent.tool` decorator, `@agent.tool_plain`, tool usage
**Prerequisites**: Lessons 01, 02
**Run**: `uv run python -m pydantic_learning.agents.03_agent_with_tools`

### 04 - Dependency Injection
**Concepts**: `deps_type`, `RunContext`, dynamic instructions
**Prerequisites**: Lessons 01, 02, 03
**Run**: `uv run python -m pydantic_learning.agents.04_dependency_injection`

### 05 - Multi-Agent Game
**Concepts**: Agent coordination, tools calling other agents, game state
**Prerequisites**: Lessons 01-04
**Run**: `uv run python -m pydantic_learning.agents.05_multi_agent_game`
**Based on**: `src/agents/twenty_questions.py`

### 06 - Multi-Agent Research
**Concepts**: Parallel execution with `asyncio.TaskGroup`, multiple agents
**Prerequisites**: Lessons 01-05
**Run**: `uv run python -m pydantic_learning.agents.06_multi_agent_research`
**Based on**: `src/agents/deep_research.py`

### 07 - Durable Execution (DBOS)
**Concepts**: `DBOSAgent` wrapper, workflow persistence, resumption
**Prerequisites**: Lessons 01-05, PostgreSQL running
**Requirements**: PostgreSQL on localhost:5432, database 'dbos'
**Run**: `uv run python -m pydantic_learning.agents.07_durable_dbos [workflow_id]`
**Based on**: `src/agents/twenty_questions_dbos.py`

### 08 - Durable Execution (Temporal)
**Concepts**: `TemporalAgent` wrapper, workflows, fault tolerance
**Prerequisites**: Lessons 01-05, Temporal server running
**Requirements**: Temporal server on localhost:7233
**Run**: `uv run python -m pydantic_learning.agents.08_durable_temporal [workflow_id]`
**Based on**: `src/agents/twenty_questions_temporal.py`

### 09 - Evaluations
**Concepts**: `pydantic_evals`, custom evaluators, model comparison
**Prerequisites**: Lessons 01-05
**Run**: `uv run python -m pydantic_learning.agents.09_evaluations`
**Based on**: `src/agents/twenty_questions_evals.py`

## Configuration

All modules use centralized configuration from `pydantic_learning.config`:

```python
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()
agent = Agent(settings.default_model, ...)
```

### Environment Variables

Configure via environment variables with `PYDANTIC_LEARNING_` prefix or standard API key variables:

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `PYDANTIC_LEARNING_DEFAULT_MODEL` - Default model (default: `openai:gpt-4o`)
- `PYDANTIC_LEARNING_TEMPORAL_HOST` - Temporal host (default: `localhost:7233`)
- `PYDANTIC_LEARNING_TASK_QUEUE` - Temporal task queue (default: `learning-tasks`)
- `PYDANTIC_LEARNING_DBOS_HOST` - DBOS PostgreSQL host (default: `localhost`)
- `PYDANTIC_LEARNING_DBOS_PORT` - DBOS PostgreSQL port (default: `5432`)
- `PYDANTIC_LEARNING_LOGFIRE_ENABLED` - Enable Logfire observability (default: `false`)

## Module Template

Each module follows this structure:

```python
"""
Lesson XX: Title
================

Brief description of what this lesson demonstrates.

Concepts Covered:
- Concept 1
- Concept 2

Prerequisites:
- Lesson YY: Previous Lesson

Run with:
    uv run python -m pydantic_learning.agents.XX_module_name
"""

import asyncio
from pydantic_ai import Agent
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()

# ============================================================
# AGENT DEFINITION
# ============================================================

agent = Agent(settings.default_model, ...)

# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    """Run the example."""
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

## Migration from Old Structure

This package consolidates:

1. **`src/agents/`** - Original standalone scripts
   - `twenty_questions.py` → Lesson 05
   - `deep_research.py` → Lesson 06
   - `*_dbos.py` → Lesson 07
   - `*_temporal.py` → Lesson 08
   - `*_evals.py` → Lesson 09

2. **`pydantic-ai-learning/backend/app/lessons/`** - Previous teaching modules
   - Concepts integrated into new numbered progression
   - Configuration pattern adopted from `settings.py`

## Development

To add a new lesson:

1. Create `XX_lesson_name.py` following the template
2. Document concepts, prerequisites, and run instructions
3. Use centralized config: `from pydantic_learning.config import get_agent_settings`
4. Update this README with the new lesson details
5. Ensure the module is self-contained and runnable

## Running the Examples

Make sure you have the required dependencies:

```bash
# Install dependencies
uv pip install pydantic-ai pydantic-evals logfire

# For DBOS examples (Lesson 07)
uv pip install dbos

# For Temporal examples (Lesson 08)
uv pip install temporalio
```

Then run any lesson:

```bash
uv run python -m pydantic_learning.agents.01_basic_agent
uv run python -m pydantic_learning.agents.02_structured_output
# ... etc
```

## Next Steps

After completing all lessons, you'll understand:

- ✅ How to create and configure PydanticAI agents
- ✅ Structured outputs with Pydantic models
- ✅ Adding tools to agents
- ✅ Dependency injection patterns
- ✅ Multi-agent coordination
- ✅ Parallel execution patterns
- ✅ Durable execution with DBOS and Temporal
- ✅ Systematic agent evaluation

You're ready to build production-grade AI agent applications!
