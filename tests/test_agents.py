"""
Tests for PydanticAI agents in src/agents/.

All tests use TestModel - no real API calls are made.
"""
import pytest
from pydantic_ai.models.test import TestModel


class TestTwentyQuestionsAgent:
    """Tests for twenty_questions.py agents."""

    @pytest.mark.asyncio
    async def test_answerer_agent_responds(self, test_model_with_structured):
        """Answerer agent should return an Answer enum value."""
        from src.agents.twenty_questions import answerer_agent, Answer

        model = test_model_with_structured(value='yes')

        with answerer_agent.override(model=model):
            result = await answerer_agent.run("Is it alive?", deps="potato")

        assert result.output is not None
        assert result.usage().requests >= 1

    @pytest.mark.asyncio
    async def test_questioner_agent_has_tool(self, test_model):
        """Questioner agent should have ask_question tool registered."""
        from src.agents.twenty_questions import questioner_agent

        # Verify the tool is registered
        tools = questioner_agent._function_tools
        assert len(tools) > 0
        assert any(tool.name == 'ask_question' for tool in tools.values())


class TestDeepResearchAgents:
    """Tests for deep_research.py agents."""

    @pytest.mark.asyncio
    async def test_plan_agent_structured_output(self, test_model_with_structured):
        """Plan agent should return DeepResearchPlan structure."""
        from src.agents.deep_research import plan_agent, DeepResearchPlan

        model = test_model_with_structured(
            executive_summary="Test summary",
            web_search_steps=[{"search_terms": "test query"}],
            analysis_instructions="Analyze results"
        )

        with plan_agent.override(model=model):
            result = await plan_agent.run("Research AI trends")

        assert isinstance(result.output, DeepResearchPlan)
        assert result.output.executive_summary == "Test summary"

    @pytest.mark.asyncio
    async def test_search_agent_responds(self, test_model):
        """Search agent should return a response."""
        from src.agents.deep_research import search_agent

        with search_agent.override(model=test_model):
            result = await search_agent.run("AI trends")

        assert result.output is not None
        assert result.usage().requests >= 1

    @pytest.mark.asyncio
    async def test_analysis_agent_has_tool(self, test_model):
        """Analysis agent should have extra_search tool registered."""
        from src.agents.deep_research import analysis_agent

        # Verify the tool is registered
        tools = analysis_agent._function_tools
        assert len(tools) > 0
        assert any(tool.name == 'extra_search' for tool in tools.values())


class TestAgentConfiguration:
    """Tests for agent configuration and setup."""

    def test_agents_have_names(self):
        """Agents should have descriptive names for observability."""
        from src.agents.deep_research import plan_agent, search_agent, analysis_agent

        assert plan_agent.name == 'abstract_plan_agent'
        assert search_agent.name == 'search_agent'
        assert analysis_agent.name == 'analysis_agent'

    def test_agents_have_output_types(self):
        """Agents with structured output should have output_type set."""
        from src.agents.deep_research import plan_agent, DeepResearchPlan
        from src.agents.twenty_questions import answerer_agent, Answer

        assert plan_agent._output_type == DeepResearchPlan
        assert answerer_agent._output_type == Answer
