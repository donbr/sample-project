"""
Lesson 04: Dependency Injection
================================

This lesson demonstrates how to use dependency injection with PydanticAI agents.

Concepts Covered:
- deps_type parameter for type-safe dependencies
- RunContext for accessing dependencies in tools
- Dynamic instructions based on dependencies

Prerequisites:
- Lesson 01: Basic Agent
- Lesson 02: Structured Output
- Lesson 03: Agent with Tools

Run with:
    uv run python -m pydantic_learning.agents.04_dependency_injection
"""

import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()


# ============================================================
# DEPENDENCIES
# ============================================================

@dataclass
class UserContext:
    """Context information about the current user."""
    name: str
    favorite_color: str
    age: int


# ============================================================
# AGENT DEFINITION
# ============================================================

agent = Agent(
    settings.default_model,
    deps_type=UserContext,
    instructions="You are a personalized assistant who uses the user's preferences.",
)


@agent.instructions
def add_user_info(ctx: RunContext[UserContext]) -> str:
    """Add dynamic instructions based on user context."""
    return f"The user's name is {ctx.deps.name} and they love the color {ctx.deps.favorite_color}."


@agent.tool
async def get_user_age(ctx: RunContext[UserContext]) -> int:
    """Get the current user's age."""
    return ctx.deps.age


@agent.tool
async def get_favorite_color(ctx: RunContext[UserContext]) -> str:
    """Get the user's favorite color."""
    return ctx.deps.favorite_color


# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    """Run the dependency injection example."""
    # Create user context
    user = UserContext(name="Alice", favorite_color="blue", age=30)

    # Run agent with dependencies
    result = await agent.run(
        "Tell me something personalized about my preferences.",
        deps=user
    )

    print(f"Response: {result.output}")
    print(f"Tool calls made: {result.usage().tool_calls}")


if __name__ == "__main__":
    asyncio.run(main())
