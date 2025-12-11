"""Lesson 01: Basic Agent

This is the simplest possible PydanticAI agent example.
It demonstrates:
- Creating an agent with a model
- Setting basic instructions
- Running a simple query

Run with: uv run python -m pydantic_learning.agents.01_basic_agent
"""

import asyncio

from pydantic_ai import Agent

from pydantic_learning.config import settings


# Create a basic agent with instructions
agent = Agent(
    settings.basic_model,
    instructions='You are a helpful assistant. Answer questions concisely.',
)


async def main():
    """Run a simple agent query."""
    result = await agent.run('What is PydanticAI?')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
