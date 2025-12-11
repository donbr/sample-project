"""
Lesson 02: Structured Output
=============================

This lesson demonstrates how to use Pydantic models for structured agent output.

Concepts Covered:
- Using output_type parameter with Pydantic models
- Defining structured response schemas
- Accessing typed output data

Prerequisites:
- Lesson 01: Basic Agent

Run with:
    uv run python -m pydantic_learning.agents.02_structured_output
"""

import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()


# ============================================================
# RESPONSE MODEL
# ============================================================

class BookRecommendation(BaseModel):
    """A book recommendation with metadata."""
    title: str
    author: str
    genre: str
    why_recommended: str
    year_published: int | None = None


# ============================================================
# AGENT DEFINITION
# ============================================================

agent = Agent(
    settings.default_model,
    instructions="You are a knowledgeable librarian who recommends books.",
    output_type=BookRecommendation,
)


# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    """Run the structured output example."""
    result = await agent.run("Recommend a science fiction book for someone new to the genre.")

    # The output is now a typed Pydantic model
    book = result.output
    print(f"Title: {book.title}")
    print(f"Author: {book.author}")
    print(f"Genre: {book.genre}")
    print(f"Why: {book.why_recommended}")
    if book.year_published:
        print(f"Published: {book.year_published}")


if __name__ == "__main__":
    asyncio.run(main())
