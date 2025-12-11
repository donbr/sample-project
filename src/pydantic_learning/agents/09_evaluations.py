"""
Lesson 09: Evaluations
=======================

This lesson demonstrates how to use pydantic_evals for testing agent performance.

Concepts Covered:
- Creating evaluation datasets
- Defining custom evaluators
- Running systematic tests across models
- Measuring agent performance

Prerequisites:
- Lesson 01: Basic Agent
- Lesson 02: Structured Output
- Lesson 03: Agent with Tools
- Lesson 04: Dependency Injection
- Lesson 05: Multi-Agent Game

Based on:
- src/agents/twenty_questions_evals.py

Run with:
    uv run python -m pydantic_learning.agents.09_evaluations
"""

import asyncio
from dataclasses import dataclass
from typing import Any, TypedDict

import logfire
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, UsageLimitExceeded
from pydantic_ai.models import KnownModelName
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_learning.config import get_agent_settings

# Import the game from lesson 05
# Note: You would typically import like this:
# from pydantic_learning.agents.05_multi_agent_game import play, questioner_agent
# However, due to Python module naming restrictions (can't start with a number),
# we need to use importlib or redefine the necessary parts here.
# For simplicity in this learning example, we'll use a relative import approach:

import sys
from pathlib import Path

# Add the agents directory to the path for importing numbered modules
agents_dir = Path(__file__).parent
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

# Now we can import (though this is a workaround for numbered module names)
# For a production setup, consider using non-numeric prefixes like 'lesson_05_...'
try:
    from importlib import import_module
    lesson_05 = import_module('05_multi_agent_game')
    play = lesson_05.play
    questioner_agent = lesson_05.questioner_agent
except ImportError:
    # Fallback: redefine what we need (not recommended, just for demonstration)
    print("Warning: Could not import lesson 05. This evaluation example requires lesson 05 to be available.")
    raise

settings = get_agent_settings()

# Configure logfire if enabled
if settings.logfire_enabled:
    logfire.configure(console=False)
    logfire.instrument_pydantic_ai()


# ============================================================
# EVALUATION RESULT TYPE
# ============================================================

class PlayResult(TypedDict):
    """Result of a game play for evaluation."""
    steps: float
    responses: list[Any]
    success: bool


# ============================================================
# CUSTOM EVALUATORS
# ============================================================

@dataclass
class QuestionCount(Evaluator[str, PlayResult]):
    """Evaluator that counts the number of questions asked."""

    async def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> float:
        """Return the number of questions asked (lower is better)."""
        return ctx.output['steps']


@dataclass
class QnASuccess(Evaluator[str, PlayResult]):
    """Evaluator that checks if the game was completed successfully."""

    async def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> bool:
        """Return whether the game completed successfully."""
        return ctx.output['success']


# ============================================================
# EVALUATION DATASET
# ============================================================

dataset: Dataset[str, PlayResult] = Dataset(
    cases=[
        Case(name='Potato', inputs='potato'),
        Case(name='Man', inputs='man'),
        Case(name='Woman', inputs='woman'),
        Case(name='Child', inputs='child'),
        Case(name='Bike', inputs='bike'),
        Case(name='House', inputs='house'),
    ],
    evaluators=[QuestionCount(), QnASuccess()],
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

async def play_eval(answer: str) -> PlayResult:
    """Run a game and return evaluation results."""
    try:
        result = await play(answer)
    except UsageLimitExceeded:
        # If we hit the usage limit, return a failed result
        return {'steps': 25, 'responses': [], 'success': False}

    # Extract responses from the result
    responses: list[Any] = []
    for message in result.all_messages():
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    responses.append(part.content)
                if isinstance(part, ToolCallPart):
                    responses.append(part.args)

    return {
        'steps': len(result.all_messages()) / 2,
        'responses': responses,
        'success': True
    }


# ============================================================
# RUN EVALUATIONS ACROSS MODELS
# ============================================================

async def run_evals():
    """Run evaluations across multiple models."""
    models: list[KnownModelName] = [
        'anthropic:claude-sonnet-4-0',
        'anthropic:claude-sonnet-4-5',
        'openai:gpt-4.1',
        'openai:gpt-4.1-mini',
        'google-vertex:gemini-2.5-flash',
    ]

    for model in models:
        print(f'\n{"="*60}')
        print(f'Evaluating model: {model}')
        print(f'{"="*60}\n')

        # Override the questioner agent's model for this evaluation
        with questioner_agent.override(model=model):
            report = await dataset.evaluate(play_eval, name=f'Q&A {model}')
            report.print(include_input=False, include_output=False)


# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    """Run the evaluations example."""
    await run_evals()


if __name__ == '__main__':
    asyncio.run(main())
