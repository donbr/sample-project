from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent
from temporalio import workflow
from ..settings import settings

# 1. Define the Standard Agent
simple_agent = Agent(
    settings.default_model,
    instructions="You are a durable agent. Acknowledge your undying nature.",
    name="simple_durable_agent"
)

# 2. Wrap it
temporal_simple_agent = TemporalAgent(simple_agent)

# 3. Define the Workflow
@workflow.defn
class SimpleDurableWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        # We call .run() on the TemporalAgent inside the workflow
        result = await temporal_simple_agent.run(f"Hello, my name is {name}")
        return result.output

# Client helper for the API
async def start_lesson1_workflow(client, name: str) -> str:
    handle = await client.execute_workflow(
        SimpleDurableWorkflow.run,
        args=[name],
        id=f"lesson1-{name}",
        task_queue=settings.task_queue
    )
    return handle
