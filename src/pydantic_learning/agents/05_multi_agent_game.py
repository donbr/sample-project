"""
Lesson 05: Multi-Agent Game
============================

This lesson demonstrates agent-to-agent communication in a Twenty Questions game.

Concepts Covered:
- Multiple agents working together
- Agent coordination pattern
- Tools that call other agents
- Game state management

Prerequisites:
- Lesson 01: Basic Agent
- Lesson 02: Structured Output
- Lesson 03: Agent with Tools
- Lesson 04: Dependency Injection

Based on:
- src/agents/twenty_questions.py

Run with:
    uv run python -m pydantic_learning.agents.05_multi_agent_game
"""

import asyncio
from dataclasses import dataclass
from enum import StrEnum

import logfire
from pydantic_ai import Agent, AgentRunResult, RunContext, UsageLimits
from pydantic_learning.config import get_agent_settings

settings = get_agent_settings()

# Configure logfire if enabled
if settings.logfire_enabled:
    logfire.configure(send_to_logfire='if-token-present', console=False)
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
# ANSWERER AGENT (knows the secret)
# ============================================================

answerer_agent = Agent(
    settings.default_model,
    deps_type=str,
    instructions="""
You are playing a question and answer game.
Your job is to answer questions about a secret object only you know truthfully.
""",
    output_type=Answer,
)


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
# QUESTIONER AGENT (tries to guess)
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
)


@questioner_agent.tool
async def ask_question(ctx: RunContext[GameState], question: str) -> Answer:
    """Ask a question to the answerer agent."""
    result = await answerer_agent.run(question, deps=ctx.deps.answer)
    print(f'{ctx.run_step:>2}: {question}: {result.output}')
    return result.output


# ============================================================
# GAME EXECUTION
# ============================================================

async def play(answer: str) -> AgentRunResult[str]:
    """Play a game of Twenty Questions."""
    state = GameState(answer=answer)
    result = await questioner_agent.run(
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
    """Run the multi-agent game example."""
    await play('potato')


if __name__ == '__main__':
    asyncio.run(main())
