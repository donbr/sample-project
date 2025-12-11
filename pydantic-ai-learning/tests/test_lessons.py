import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic_ai.models.test import TestModel

# Import the new function names
from backend.app.lessons.lesson1_temporal_basics import simple_agent, start_lesson1_workflow, SimpleDurableWorkflow
from backend.app.lessons.lesson2_deep_research import planner, searcher, analyst, start_lesson2_workflow, DeepResearchWorkflow
from backend.app.lessons.lesson3_human_loop import guesser, start_lesson3_workflow, GameWorkflow

# We need to mock the Temporal Client
@pytest.fixture
def mock_temporal_client():
    client = AsyncMock()
    # Mock execute_workflow to return a handle with a result
    async def mock_execute(workflow, args, **kwargs):
        handle = AsyncMock()
        
        # Here is the trick: We want to actually RUN the workflow logic to test the agents.
        # But running Temporal workflows locally without a server is hard because of 'workflow.*' calls.
        # So for UNIT tests, we usually just mock the workflow execution and trust the agent logic.
        
        # HOWEVER, to test the agents themselves:
        # We can instantiate the workflow class and run it? 
        # No, because it calls 'temporal_agent.run()' which expects a workflow context.
        
        # So, broadly, unit testing Temporal Workflows requires 'temporalio.testing'.
        # Since we don't have that setup, we will just mock the return value to verify "wiring".
        # The actual "Agent" logic is tested by overriding the agents and manually running them 
        # if we could, but here they are wrapped.
        
        # Let's just return a success string relative to the lesson
        if workflow == SimpleDurableWorkflow.run:
            handle.result.return_value = f"Hello, my name is {args[0]}"
        elif workflow == DeepResearchWorkflow.run:
             handle.result.return_value = "Final Report"
        elif workflow == GameWorkflow.run:
             handle.result.return_value = "Q1: ... -> Q2: ..."
             
        return handle

    client.execute_workflow.side_effect = mock_execute
    return client

@pytest.mark.asyncio
async def test_lesson1_temporal(mock_temporal_client):
    # This test primarily verifies the client-side calling convention
    result = await start_lesson1_workflow(mock_temporal_client, "Tester")
    assert result == "Hello, my name is Tester"

@pytest.mark.asyncio
async def test_lesson2_deep_research(mock_temporal_client):
    result = await start_lesson2_workflow(mock_temporal_client, "AI")
    assert result == "Final Report"

@pytest.mark.asyncio
async def test_lesson3_human_loop(mock_temporal_client):
    result = await start_lesson3_workflow(mock_temporal_client, "Animal")
    assert "Q1" in result
