"""Unit tests for AgentExecutor.

Tests cover:
- Prompt rendering with context
- Provider execution
- Output validation
- Error handling
"""

import pytest

from copilot_conductor.config.schema import AgentDef, OutputField
from copilot_conductor.exceptions import TemplateError, ValidationError
from copilot_conductor.executor.agent import AgentExecutor
from copilot_conductor.providers.base import AgentOutput
from copilot_conductor.providers.copilot import CopilotProvider


@pytest.fixture
def simple_agent() -> AgentDef:
    """Create a simple agent definition."""
    return AgentDef(
        name="test_agent",
        model="gpt-4",
        prompt="Answer the question: {{ workflow.input.question }}",
        output={"answer": OutputField(type="string")},
    )


@pytest.fixture
def agent_with_system_prompt() -> AgentDef:
    """Create an agent with system prompt."""
    return AgentDef(
        name="test_agent",
        model="gpt-4",
        system_prompt="You are a helpful assistant for {{ workflow.input.topic }}.",
        prompt="Answer: {{ workflow.input.question }}",
        output={"answer": OutputField(type="string")},
    )


@pytest.fixture
def agent_without_output_schema() -> AgentDef:
    """Create an agent without output schema."""
    return AgentDef(
        name="test_agent",
        model="gpt-4",
        prompt="Do something",
        output=None,
    )


class TestAgentExecutorBasic:
    """Basic AgentExecutor tests."""

    @pytest.mark.asyncio
    async def test_execute_renders_prompt(self, simple_agent: AgentDef) -> None:
        """Test that execute renders the prompt template."""
        received_prompts = []

        def mock_handler(agent, prompt, context):
            received_prompts.append(prompt)
            return {"answer": "Python is great"}

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        context = {"workflow": {"input": {"question": "What is Python?"}}}
        await executor.execute(simple_agent, context)

        assert len(received_prompts) == 1
        assert "What is Python?" in received_prompts[0]

    @pytest.mark.asyncio
    async def test_execute_returns_output(self, simple_agent: AgentDef) -> None:
        """Test that execute returns the agent output."""

        def mock_handler(agent, prompt, context):
            return {"answer": "The answer is 42"}

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        context = {"workflow": {"input": {"question": "What is the answer?"}}}
        output = await executor.execute(simple_agent, context)

        assert isinstance(output, AgentOutput)
        assert output.content["answer"] == "The answer is 42"

    @pytest.mark.asyncio
    async def test_execute_validates_output(self, simple_agent: AgentDef) -> None:
        """Test that execute validates output against schema."""

        def mock_handler(agent, prompt, context):
            return {"answer": 42}  # Wrong type - should be string

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        context = {"workflow": {"input": {"question": "test"}}}

        with pytest.raises(ValidationError, match="wrong type"):
            await executor.execute(simple_agent, context)

    @pytest.mark.asyncio
    async def test_execute_without_schema_skips_validation(
        self, agent_without_output_schema: AgentDef
    ) -> None:
        """Test that execute skips validation when no schema defined."""

        def mock_handler(agent, prompt, context):
            return {"anything": "goes", "here": 123}

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        context = {"workflow": {"input": {}}}
        output = await executor.execute(agent_without_output_schema, context)

        assert output.content["anything"] == "goes"
        assert output.content["here"] == 123


class TestAgentExecutorPromptRendering:
    """Tests for prompt rendering."""

    @pytest.mark.asyncio
    async def test_render_prompt_with_nested_context(self) -> None:
        """Test rendering prompt with nested context values."""
        agent = AgentDef(
            name="test",
            model="gpt-4",
            prompt="Plan: {{ planner.output.plan }}\nQuestion: {{ workflow.input.question }}",
            output=None,
        )

        def mock_handler(agent, prompt, context):
            return {"result": "ok"}

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        context = {
            "workflow": {"input": {"question": "How?"}},
            "planner": {"output": {"plan": "Step 1, Step 2"}},
        }
        await executor.execute(agent, context)

        # Verify prompt was rendered (via call history)
        call_history = provider.get_call_history()
        assert "Step 1, Step 2" in call_history[0]["prompt"]
        assert "How?" in call_history[0]["prompt"]

    @pytest.mark.asyncio
    async def test_render_prompt_with_json_filter(self) -> None:
        """Test rendering prompt with json filter."""
        agent = AgentDef(
            name="test",
            model="gpt-4",
            prompt="Data: {{ data | json }}",
            output=None,
        )

        def mock_handler(agent, prompt, context):
            return {"result": "ok"}

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        context = {"data": {"key": "value", "items": [1, 2, 3]}}
        await executor.execute(agent, context)

        call_history = provider.get_call_history()
        # JSON should be in the prompt
        assert '"key"' in call_history[0]["prompt"]
        assert '"value"' in call_history[0]["prompt"]

    @pytest.mark.asyncio
    async def test_render_prompt_missing_variable_raises(self) -> None:
        """Test that missing template variable raises TemplateError."""
        agent = AgentDef(
            name="test",
            model="gpt-4",
            prompt="Value: {{ missing.variable }}",
            output=None,
        )

        def mock_handler(agent, prompt, context):
            return {"result": "ok"}

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        context = {}

        with pytest.raises(TemplateError, match="Undefined variable"):
            await executor.execute(agent, context)

    def test_render_prompt_helper(self, simple_agent: AgentDef) -> None:
        """Test the render_prompt helper method."""
        provider = CopilotProvider()
        executor = AgentExecutor(provider)

        context = {"workflow": {"input": {"question": "Test question?"}}}
        rendered = executor.render_prompt(simple_agent, context)

        assert "Test question?" in rendered


class TestAgentExecutorWithTools:
    """Tests for agent execution with tools."""

    @pytest.mark.asyncio
    async def test_execute_passes_tools_to_provider(self) -> None:
        """Test that tools are passed to the provider."""
        agent = AgentDef(
            name="test",
            model="gpt-4",
            prompt="Use tools",
            tools=["web_search", "calculator"],
            output=None,
        )

        def mock_handler(agent, prompt, context):
            return {"result": "ok"}

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        await executor.execute(agent, {})

        call_history = provider.get_call_history()
        assert call_history[0]["tools"] == ["web_search", "calculator"]

    @pytest.mark.asyncio
    async def test_execute_with_no_tools(self) -> None:
        """Test execution with no tools specified."""
        agent = AgentDef(
            name="test",
            model="gpt-4",
            prompt="No tools",
            tools=None,
            output=None,
        )

        def mock_handler(agent, prompt, context):
            return {"result": "ok"}

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        await executor.execute(agent, {})

        call_history = provider.get_call_history()
        assert call_history[0]["tools"] is None

    @pytest.mark.asyncio
    async def test_execute_with_empty_tools(self) -> None:
        """Test execution with empty tools list."""
        agent = AgentDef(
            name="test",
            model="gpt-4",
            prompt="No tools allowed",
            tools=[],
            output=None,
        )

        def mock_handler(agent, prompt, context):
            return {"result": "ok"}

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        await executor.execute(agent, {})

        call_history = provider.get_call_history()
        assert call_history[0]["tools"] == []


class TestAgentExecutorOutputHandling:
    """Tests for output handling edge cases."""

    @pytest.mark.asyncio
    async def test_missing_output_field_raises(self) -> None:
        """Test that missing required output field raises ValidationError."""
        agent = AgentDef(
            name="test",
            model="gpt-4",
            prompt="Test",
            output={
                "required_field": OutputField(type="string"),
                "another_field": OutputField(type="number"),
            },
        )

        def mock_handler(agent, prompt, context):
            return {"required_field": "value"}  # Missing another_field

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        with pytest.raises(ValidationError, match="Missing required output field"):
            await executor.execute(agent, {})

    @pytest.mark.asyncio
    async def test_output_with_multiple_types(self) -> None:
        """Test validation of output with multiple field types."""
        agent = AgentDef(
            name="test",
            model="gpt-4",
            prompt="Test",
            output={
                "text": OutputField(type="string"),
                "count": OutputField(type="number"),
                "active": OutputField(type="boolean"),
                "items": OutputField(type="array"),
                "meta": OutputField(type="object"),
            },
        )

        def mock_handler(agent, prompt, context):
            return {
                "text": "hello",
                "count": 42,
                "active": True,
                "items": [1, 2, 3],
                "meta": {"key": "value"},
            }

        provider = CopilotProvider(mock_handler=mock_handler)
        executor = AgentExecutor(provider)

        output = await executor.execute(agent, {})

        assert output.content["text"] == "hello"
        assert output.content["count"] == 42
        assert output.content["active"] is True
        assert output.content["items"] == [1, 2, 3]
        assert output.content["meta"] == {"key": "value"}
