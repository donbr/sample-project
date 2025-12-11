"""
Lesson 01: Basic Agent
=======================

This lesson demonstrates the minimal setup for a PydanticAI agent.

Concepts Covered:
- Creating a basic Agent instance
- Running an agent with a simple prompt
- Understanding agent output

Prerequisites:
- None (this is the starting point)

Run with:
    uv run python -m pydantic_learning.agents.01_basic_agent
"""

import asyncio
from pydantic_ai import Agent
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()

# ============================================================
# AGENT DEFINITION
# ============================================================

agent = Agent(
    settings.default_model,
    instructions="You are a helpful assistant.",
)


# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    """Run the basic agent example."""
    result = await agent.run("Hello! What can you help me with?")
    print(f"Agent response: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
