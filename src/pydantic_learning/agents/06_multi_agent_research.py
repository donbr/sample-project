"""
Lesson 06: Multi-Agent Research
================================

This lesson demonstrates parallel agent execution for deep research tasks.

Concepts Covered:
- Parallel agent execution with asyncio.TaskGroup
- Multi-agent coordination pattern
- Planning, searching, and analysis agents
- Web search integration

Prerequisites:
- Lesson 01: Basic Agent
- Lesson 02: Structured Output
- Lesson 03: Agent with Tools
- Lesson 04: Dependency Injection
- Lesson 05: Multi-Agent Game

Based on:
- src/agents/deep_research.py

Run with:
    uv run python -m pydantic_learning.agents.06_multi_agent_research
"""

import asyncio
from typing import Annotated

import logfire
from annotated_types import MaxLen
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, RunContext, WebSearchTool, format_as_xml
from pydantic_ai.agent import AbstractAgent
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()

# Configure logfire if enabled
if settings.logfire_enabled:
    logfire.configure()
    logfire.instrument_pydantic_ai()


# ============================================================
# RESEARCH PLAN MODEL
# ============================================================

class WebSearchStep(BaseModel):
    """A step that performs a web search and returns a summary."""
    search_terms: str


class DeepResearchPlan(BaseModel, **ConfigDict(use_attribute_docstrings=True)):
    """A structured plan for deep research."""

    executive_summary: str
    """A summary of the research plan."""

    web_search_steps: Annotated[list[WebSearchStep], MaxLen(5)]
    """A list of web search steps to perform to gather raw information."""

    analysis_instructions: str
    """The analysis step to perform after all web search steps are completed."""


# ============================================================
# PLANNING AGENT
# ============================================================

plan_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Analyze the users query and design a plan for deep research to answer their query.',
    output_type=DeepResearchPlan,
    name='plan_agent',
)


# ============================================================
# SEARCH AGENT
# ============================================================

search_agent = Agent(
    'google-vertex:gemini-2.5-flash',
    instructions='Perform a web search for the given terms and return a detailed report on the results.',
    builtin_tools=[WebSearchTool()],
    name='search_agent',
)


# ============================================================
# ANALYSIS AGENT
# ============================================================

analysis_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    deps_type=AbstractAgent,
    instructions="""
Analyze the research from the previous steps and generate a report on the given subject.

If the search results do not contain enough information, you may perform further searches using the
`extra_search` tool.
""",
    name='analysis_agent',
)


@analysis_agent.tool
async def extra_search(ctx: RunContext[AbstractAgent], query: str) -> str:
    """Perform an extra search for the given query."""
    result = await ctx.deps.run(query)
    return result.output


# ============================================================
# RESEARCH ORCHESTRATION
# ============================================================

@logfire.instrument
async def deep_research(query: str) -> str:
    """Execute a deep research task using multiple agents in parallel."""
    # Step 1: Create a research plan
    result = await plan_agent.run(query)
    plan = result.output

    # Step 2: Execute all search steps in parallel
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(search_agent.run(step.search_terms))
            for step in plan.web_search_steps
        ]

    # Collect search results
    search_results = [task.result().output for task in tasks]

    # Step 3: Analyze all results
    analysis_result = await analysis_agent.run(
        format_as_xml({
            'query': query,
            'search_results': search_results,
            'instructions': plan.analysis_instructions,
        }),
        deps=search_agent,
    )

    return analysis_result.output


# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    """Run the multi-agent research example."""
    result = await deep_research('Find me a list of hedge funds that write python in London')
    print(f"Research Result:\n{result}")


if __name__ == '__main__':
    asyncio.run(main())
