"""Lesson 02: Agent with Tools

This example demonstrates how to give an agent tools (functions) it can call.
It demonstrates:
- Defining tools with the @agent.tool decorator
- Tool parameters and return types
- How agents decide when to use tools

Run with: uv run python -m pydantic_learning.agents.02_agent_with_tools
"""

import asyncio
from datetime import datetime

from pydantic_ai import Agent, RunContext


agent = Agent(
    'anthropic:claude-3-5-haiku-latest',
    instructions='You are a helpful assistant with access to utility tools.',
)


@agent.tool
def get_current_time(ctx: RunContext) -> str:
    """Get the current time in ISO format."""
    return datetime.now().isoformat()


@agent.tool
def add_numbers(ctx: RunContext, a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


async def main():
    """Run queries that demonstrate tool usage."""
    # Query that will use the time tool
    result1 = await agent.run('What time is it right now?')
    print(f"Time query: {result1.output}\n")

    # Query that will use the math tool
    result2 = await agent.run('What is 42 plus 17?')
    print(f"Math query: {result2.output}")


if __name__ == '__main__':
    asyncio.run(main())
