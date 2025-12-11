from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent
from temporalio import workflow
from ..settings import settings

guesser = Agent(
    settings.default_model,
    instructions="You are playing a guessing game. Ask a yes/no question to identify the secret object.",
    name="guesser"
)

wf_guesser = TemporalAgent(guesser)

@workflow.defn
class GameWorkflow:
    @workflow.run
    async def run(self, secret_category: str) -> str:
        # Turn 1
        q1 = await wf_guesser.run(f"I am thinking of a {secret_category}. Ask me a question.")
        
        # In a real app, typically you wait for a signal:
        # await workflow.wait_condition(lambda: self.user_input is not None)
        # For this lesson, we simulate the 'next turn' simply:
        
        q2 = await wf_guesser.run(
            f"Previous Question: {q1.output}\nAnswer: Yes.\nAsk another question."
        )
        
        return f"Q1: {q1.output} (Yes) -> Q2: {q2.output}"

async def start_lesson3_workflow(client, category: str) -> str:
    handle = await client.execute_workflow(
        GameWorkflow.run,
        args=[category],
        id=f"lesson3-{category}",
        task_queue=settings.task_queue
    )
    return handle
