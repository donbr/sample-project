import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent
from temporalio import workflow
from ..settings import settings

# Data Models
class SearchStep(BaseModel):
    query: str
    reason: str

class ResearchPlan(BaseModel):
    steps: list[SearchStep]

# Agents
planner = Agent(
    settings.default_model,
    output_type=ResearchPlan,
    instructions="Generate 3 search queries to research the given topic.",
    name="planner"
)

searcher = Agent(
    settings.default_model,
    instructions="Simulate a search engine. Return fake results for the query.",
    name="searcher"
)

analyst = Agent(
    settings.default_model,
    instructions="Summarize the search results into a final answer.",
    name="analyst"
)

# Temporal Wrappers
wf_planner = TemporalAgent(planner)
wf_searcher = TemporalAgent(searcher)
wf_analyst = TemporalAgent(analyst)

@workflow.defn
class DeepResearchWorkflow:
    @workflow.run
    async def run(self, topic: str) -> str:
        # Step 1: Plan
        plan_result = await wf_planner.run(f"Research this: {topic}")
        plan = plan_result.output
        
        # Step 2: Execute Parallel Searches
        search_results = []
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(wf_searcher.run(step.query))
                for step in plan.steps
            ]
        
        # Gather results
        results_text = "\n".join([t.result().output for t in tasks])
        
        # Step 3: Analyze
        final_result = await wf_analyst.run(
            f"Topic: {topic}\nResults:\n{results_text}"
        )
        return final_result.output

async def start_lesson2_workflow(client, topic: str) -> str:
    handle = await client.execute_workflow(
        DeepResearchWorkflow.run,
        args=[topic],
        id=f"lesson2-{topic}",
        task_queue=settings.task_queue
    )
    return handle
