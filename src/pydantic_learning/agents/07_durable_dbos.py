"""
Lesson 07: Durable Execution with DBOS
=======================================

This lesson demonstrates how to make agents durable using DBOS integration.

Concepts Covered:
- DBOSAgent wrapper for durability
- Workflow persistence and resumption
- Database-backed execution tracking
- Fault tolerance

Prerequisites:
- Lesson 01: Basic Agent
- Lesson 02: Structured Output
- Lesson 03: Agent with Tools
- Lesson 04: Dependency Injection
- Lesson 05: Multi-Agent Game

Based on:
- src/agents/twenty_questions_dbos.py

Requirements:
- PostgreSQL running on localhost:5432
- Database 'dbos' created

Run with:
    uv run python -m pydantic_learning.agents.07_durable_dbos [workflow_id]

If workflow_id is provided, it will resume that workflow.
"""

import asyncio
import sys
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import logfire
from dbos import DBOS, DBOSConfig, SetWorkflowID, WorkflowHandle
from pydantic_ai import Agent, AgentRunResult, RunContext, UsageLimits
from pydantic_ai.durable_exec.dbos import DBOSAgent
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()

# Configure logfire if enabled
if settings.logfire_enabled:
    logfire.configure(console=False)
    logfire.instrument_pydantic_ai()


# ============================================================
# RESPONSE MODEL
# ============================================================

class Answer(StrEnum):
    """Possible answers to yes/no questions."""
    yes = 'yes'
    kind_of = 'kind of'
    not_really = 'not really'
    no = 'no'
    complete_wrong = 'complete wrong'


# ============================================================
# ANSWERER AGENT (wrapped with DBOS)
# ============================================================

answerer_agent = Agent(
    settings.default_model,
    deps_type=str,
    instructions="""
You are playing a question and answer game.
Your job is to answer questions about a secret object only you know truthfully.
""",
    output_type=Answer,
    name='answerer_agent',
)

# Wrap the agent with DBOS for durability
dbos_answerer_agent = DBOSAgent(answerer_agent)


@answerer_agent.instructions
def add_answer(ctx: RunContext[str]) -> str:
    """Add the secret object to the answerer's context."""
    return f'THE SECRET OBJECT IS: "{ctx.deps}".'


# ============================================================
# GAME STATE
# ============================================================

@dataclass
class GameState:
    """State for the Twenty Questions game."""
    answer: str


# ============================================================
# QUESTIONER AGENT (wrapped with DBOS)
# ============================================================

questioner_agent = Agent(
    settings.default_model,
    deps_type=GameState,
    instructions="""
You are playing a question and answer game. You need to guess what object the other player is thinking of.
Your job is to ask quantitative questions to narrow down the possibilities.

Start with broad questions (e.g., "Is it alive?", "Is it bigger than a breadbox?") and get more specific.
When you're confident, make a guess by saying "Is it [specific object]?"

You should ask strategic questions based on the previous answers.
""",
    name='questioner_agent',
)


@questioner_agent.tool
async def ask_question(ctx: RunContext[GameState], question: str) -> Answer:
    """Ask a question to the answerer agent (durable call)."""
    result = await dbos_answerer_agent.run(question, deps=ctx.deps.answer)
    print(f'{ctx.run_step:>2}: {question}: {result.output}')
    return result.output


# Wrap the questioner agent with DBOS for durability
dbos_questioner_agent = DBOSAgent(questioner_agent)


# ============================================================
# GAME EXECUTION WITH DBOS
# ============================================================

async def play(resume_id: str | None, answer: str) -> AgentRunResult[str]:
    """Play a durable game of Twenty Questions with DBOS."""
    # Configure DBOS
    config: DBOSConfig = {
        'name': 'twenty_questions_durable',
        'enable_otlp': True,
        'system_database_url': f'postgresql://postgres@{settings.dbos_host}:{settings.dbos_port}/dbos',
    }
    DBOS(config=config)
    DBOS.launch()

    if resume_id is not None:
        # Resume an existing workflow
        print(f'Resuming existing workflow: {resume_id}')
        wf: WorkflowHandle[Any] = DBOS.retrieve_workflow(resume_id)
        result = await wf.get_result()
    else:
        # Start a new workflow
        wf_id = f'twenty-questions-{uuid.uuid4()}'
        print(f'Starting new workflow: {wf_id}')

        state = GameState(answer=answer)
        with SetWorkflowID(wf_id):
            result = await dbos_questioner_agent.run(
                'start',
                deps=state,
                usage_limits=UsageLimits(request_limit=25)
            )

    print(f'After {len(result.all_messages()) / 2} questions, the answer is: {result.output}')
    return result


# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    """Run the durable DBOS example."""
    resume_id = sys.argv[1] if len(sys.argv) > 1 else None
    await play(resume_id, 'potato')


if __name__ == '__main__':
    asyncio.run(main())
