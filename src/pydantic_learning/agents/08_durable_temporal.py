"""
Lesson 08: Durable Execution with Temporal
===========================================

This lesson demonstrates how to make agents durable using Temporal integration.

Concepts Covered:
- TemporalAgent wrapper for durability
- Workflow definitions with @workflow.defn
- Temporal workers and clients
- Workflow resumption after failures

Prerequisites:
- Lesson 01: Basic Agent
- Lesson 02: Structured Output
- Lesson 03: Agent with Tools
- Lesson 04: Dependency Injection
- Lesson 05: Multi-Agent Game

Based on:
- src/agents/twenty_questions_temporal.py

Requirements:
- Temporal server running on localhost:7233

Run with:
    uv run python -m pydantic_learning.agents.08_durable_temporal [workflow_id]

If workflow_id is provided, it will resume that workflow.
"""

import asyncio
import sys
import uuid
from random import random
from typing import Literal

import logfire
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import AgentPlugin, LogfirePlugin, PydanticAIPlugin, TemporalAgent
from pydantic_ai.tools import RunContext
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()

# Configure logfire if enabled
if settings.logfire_enabled:
    logfire.configure(console=False)
    logfire.instrument_pydantic_ai()


# ============================================================
# SECRET (hardcoded for this example)
# ============================================================

secret = 'potato'


# ============================================================
# ANSWERER AGENT (wrapped with Temporal)
# ============================================================

answerer_agent = Agent(
    settings.default_model,
    instructions=f"""
You are playing a question and answer game.
Your job is to answer yes/no questions about the secret object truthfully.
ALWAYS answer with only true for 'yes' and false for 'no'.

THE SECRET OBJECT IS: {secret}.
""",
    output_type=bool,
    name='answerer_agent',
)

# Wrap the agent with Temporal for durability
temporal_answerer_agent = TemporalAgent(answerer_agent)


# ============================================================
# QUESTIONER AGENT (wrapped with Temporal)
# ============================================================

questioner_agent = Agent(
    settings.default_model,
    instructions="""
You are playing a question and answer game. You need to guess what object the other player is thinking of.
Your job is to ask yes/no questions to narrow down the possibilities.

Start with broad questions (e.g., "Is it alive?", "Is it bigger than a breadbox?") and get more specific.
When you're confident, make a guess by saying "Is it [specific object]?"

You should ask strategic questions based on the previous answers.
""",
    name='questioner_agent',
)


@questioner_agent.tool
async def ask_question(ctx: RunContext, question: str) -> Literal['yes', 'no']:
    """Ask a question to the answerer agent (durable call with simulated failures)."""
    # Simulate random failures to demonstrate durability
    if random() > 0.9:
        raise RuntimeError('Simulated failure - Temporal will retry')

    print(f'{ctx.run_step:>2}: {question}:', end=' ', flush=True)
    result = await temporal_answerer_agent.run(question)
    ans = 'yes' if result.output else 'no'
    print(ans)
    return ans


# Wrap the questioner agent with Temporal for durability
temporal_questioner_agent = TemporalAgent(questioner_agent)


# ============================================================
# WORKFLOW DEFINITION
# ============================================================

@workflow.defn
class TwentyQuestionsWorkflow:
    """Temporal workflow for the Twenty Questions game."""

    @workflow.run
    async def run(self) -> None:
        """Run the game workflow."""
        result = await temporal_questioner_agent.run('start')
        print(f'After {len(result.all_messages()) / 2} questions, the answer is: {result.output}')


# ============================================================
# GAME EXECUTION WITH TEMPORAL
# ============================================================

async def play(resume_id: str | None):
    """Play a durable game of Twenty Questions with Temporal."""
    # Connect to Temporal server
    client = await Client.connect(
        settings.temporal_host,
        plugins=[PydanticAIPlugin(), LogfirePlugin()]
    )

    # Create a worker with the workflow and agent plugins
    async with Worker(
        client,
        task_queue=settings.task_queue,
        workflows=[TwentyQuestionsWorkflow],
        plugins=[
            AgentPlugin(temporal_answerer_agent),
            AgentPlugin(temporal_questioner_agent)
        ],
    ):
        if resume_id is not None:
            # Resume an existing workflow
            print(f'Resuming existing workflow: {resume_id}')
            await client.get_workflow_handle(resume_id).result()  # type: ignore[ReportUnknownMemberType]
        else:
            # Start a new workflow
            workflow_id = f'twenty_questions-{uuid.uuid4()}'
            print(f'Starting new workflow: {workflow_id}')
            await client.execute_workflow(
                TwentyQuestionsWorkflow.run,
                id=workflow_id,
                task_queue=settings.task_queue,
            )


# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    """Run the durable Temporal example."""
    resume_id = sys.argv[1] if len(sys.argv) > 1 else None
    await play(resume_id)


if __name__ == '__main__':
    asyncio.run(main())
