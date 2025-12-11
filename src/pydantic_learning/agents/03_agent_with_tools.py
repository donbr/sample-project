"""
Lesson 03: Agent with Tools
============================

This lesson demonstrates how to add tools to a PydanticAI agent.

Concepts Covered:
- @agent.tool decorator for tools with RunContext
- @agent.tool_plain for simple, context-free tools
- Tool return types and usage

Prerequisites:
- Lesson 01: Basic Agent
- Lesson 02: Structured Output

Run with:
    uv run python -m pydantic_learning.agents.03_agent_with_tools
"""

import asyncio
from datetime import datetime
from pydantic_ai import Agent
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()


# ============================================================
# AGENT DEFINITION
# ============================================================

agent = Agent(
    settings.default_model,
    instructions="You are a helpful assistant with access to time and calculation tools.",
)


@agent.tool_plain
def get_current_time() -> str:
    """Get the current time in ISO format."""
    return datetime.now().isoformat()


@agent.tool_plain
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    """Run the agent with tools example."""
    # Example 1: Using time tool
    result1 = await agent.run("What time is it right now?")
    print(f"Response: {result1.output}")
    print(f"Tool calls made: {result1.usage().tool_calls}")
    print()

    # Example 2: Using calculation tool
    result2 = await agent.run("What is 42 plus 17?")
    print(f"Response: {result2.output}")
    print(f"Tool calls made: {result2.usage().tool_calls}")


if __name__ == "__main__":
    asyncio.run(main())
