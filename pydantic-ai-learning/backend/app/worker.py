import asyncio
from temporalio.client import Client
from temporalio.worker import Worker, UnsandboxedWorkflowRunner
from pydantic_ai.durable_exec.temporal import AgentPlugin, LogfirePlugin, PydanticAIPlugin

# Import ALL workflows and agents that need to be registered
from .lessons.lesson1_temporal_basics import SimpleDurableWorkflow, temporal_simple_agent
from .lessons.lesson2_deep_research import DeepResearchWorkflow, wf_planner, wf_searcher, wf_analyst
from .lessons.lesson3_human_loop import GameWorkflow, wf_guesser
from .settings import settings

async def run_worker():
    print(f"Connecting to Temporal at {settings.temporal_host}...")
    
    # Enable Logfire (optional if configured)
    if settings.logfire_enabled:
        logfire.configure()
        logfire.instrument_pydantic_ai()

    client = await Client.connect(
        settings.temporal_host,
        plugins=[PydanticAIPlugin(), LogfirePlugin()] if settings.logfire_enabled else [PydanticAIPlugin()]
    )

    print(f"Starting worker on queue: {settings.task_queue}")
    
    # Register Workflows and Agent Plugins
    async with Worker(
        client,
        task_queue=settings.task_queue,
        workflows=[
            SimpleDurableWorkflow, 
            DeepResearchWorkflow,
            GameWorkflow
        ],
        plugins=[
            AgentPlugin(temporal_simple_agent),
            AgentPlugin(wf_planner),
            AgentPlugin(wf_searcher),
            AgentPlugin(wf_analyst),
            AgentPlugin(wf_guesser)
        ],
        workflow_runner=UnsandboxedWorkflowRunner()
    ):
        print("Worker running. Press Ctrl+C to stop.")
        # Wait indefinitely
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(run_worker())
