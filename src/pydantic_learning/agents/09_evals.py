"""Lesson 09: Evaluation Patterns

This example demonstrates how to evaluate agent performance.
It demonstrates:
- Using pydantic-evals for systematic testing
- Creating evaluation datasets
- Custom evaluators for metrics
- Testing across multiple models

Run with: uv run python -m pydantic_learning.agents.09_evals
"""

import asyncio
from dataclasses import dataclass
from typing import Any, TypedDict

import logfire
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, UsageLimitExceeded
from pydantic_ai.models import KnownModelName
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

# Import the multi-agent example to evaluate
# Note: Python modules can't start with numbers, so we use importlib
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("multi_agent", __file__.replace("09_evals.py", "03_multi_agent.py"))
multi_agent = importlib.util.module_from_spec(spec)
sys.modules["multi_agent"] = multi_agent
spec.loader.exec_module(multi_agent)
play = multi_agent.play
questioner_agent = multi_agent.questioner_agent

logfire.configure(console=False)
logfire.instrument_pydantic_ai()


class PlayResult(TypedDict):
    steps: float
    responses: list[Any]
    success: bool


@dataclass
class QuestionCount(Evaluator[str, PlayResult]):
    async def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> float:
        return ctx.output['steps']


@dataclass
class QnASuccess(Evaluator[str, PlayResult]):
    async def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> bool:
        return ctx.output['success']


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


async def play_eval(answer: str) -> PlayResult:
    try:
        result = await play(answer)
    except UsageLimitExceeded:
        return {'steps': 25, 'responses': [], 'success': False}
    responses: list[Any] = []
    for message in result.all_messages():
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    responses.append(part.content)
                if isinstance(part, ToolCallPart):
                    responses.append(part.args)
    return {'steps': len(result.all_messages()) / 2, 'responses': responses, 'success': True}


async def run_evals():
    models: list[KnownModelName] = [
        'anthropic:claude-sonnet-4-0',
        'anthropic:claude-sonnet-4-5',
        'openai:gpt-4.1',
        'openai:gpt-4.1-mini',
        'google-vertex:gemini-2.5-flash',
    ]
    for model in models:
        with questioner_agent.override(model=model):
            report = await dataset.evaluate(play_eval, name=f'Q&A {model}')
            report.print(include_input=False, include_output=False)


if __name__ == '__main__':
    asyncio.run(run_evals())
