"""Unit tests for the CopilotProvider implementation."""

import pytest

from copilot_conductor.config.schema import AgentDef
from copilot_conductor.providers.copilot import CopilotProvider


class TestCopilotProvider:
    """Tests for the CopilotProvider class."""

    @pytest.mark.asyncio
    async def test_validate_connection(self) -> None:
        """Test that validate_connection returns True (stub behavior)."""
        provider = CopilotProvider()
        result = await provider.validate_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test that close cleans up the client."""
        provider = CopilotProvider()
        provider._client = "some_client"  # Simulate having a client
        await provider.close()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_execute_returns_stub_output(self) -> None:
        """Test that execute returns a stub AgentOutput."""
        provider = CopilotProvider()
        agent = AgentDef(
            name="test_agent",
            model="gpt-4",
            prompt="Test prompt",
        )
        result = await provider.execute(
            agent=agent,
            context={"workflow": {"input": {}}},
            rendered_prompt="Test prompt",
        )
        assert result.content == {"result": "stub response"}
        assert result.model == "gpt-4"
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_execute_uses_agent_model(self) -> None:
        """Test that execute uses the model from agent definition."""
        provider = CopilotProvider()
        agent = AgentDef(
            name="test_agent",
            model="claude-3",
            prompt="Test prompt",
        )
        result = await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Test prompt",
        )
        assert result.model == "claude-3"

    @pytest.mark.asyncio
    async def test_execute_with_no_model(self) -> None:
        """Test that execute handles agent without model."""
        provider = CopilotProvider()
        # Create agent with type="human_gate" to bypass model requirement
        # or just set model to None directly
        agent = AgentDef(
            name="test_agent",
            prompt="Test prompt",
            model=None,
        )
        result = await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Test prompt",
        )
        assert result.model == "unknown"


class TestCopilotProviderToolsSupport:
    """Tests for tool support in CopilotProvider."""

    @pytest.mark.asyncio
    async def test_execute_records_tools_in_call_history(self) -> None:
        """Test that tools are recorded in call history."""
        provider = CopilotProvider()
        agent = AgentDef(
            name="test_agent",
            model="gpt-4",
            prompt="Test prompt",
        )

        tools = ["web_search", "calculator"]
        await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Test prompt",
            tools=tools,
        )

        call_history = provider.get_call_history()
        assert len(call_history) == 1
        assert call_history[0]["tools"] == ["web_search", "calculator"]

    @pytest.mark.asyncio
    async def test_execute_with_empty_tools_list(self) -> None:
        """Test that empty tools list is recorded correctly."""
        provider = CopilotProvider()
        agent = AgentDef(
            name="test_agent",
            model="gpt-4",
            prompt="Test prompt",
        )

        await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Test prompt",
            tools=[],
        )

        call_history = provider.get_call_history()
        assert call_history[0]["tools"] == []

    @pytest.mark.asyncio
    async def test_execute_with_none_tools(self) -> None:
        """Test that None tools is recorded correctly."""
        provider = CopilotProvider()
        agent = AgentDef(
            name="test_agent",
            model="gpt-4",
            prompt="Test prompt",
        )

        await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Test prompt",
            tools=None,
        )

        call_history = provider.get_call_history()
        assert call_history[0]["tools"] is None

    @pytest.mark.asyncio
    async def test_mock_handler_receives_correct_tools(self) -> None:
        """Test that mock handler can verify tools passed to provider."""

        def mock_handler(agent, prompt, context):
            return {"result": "ok"}

        provider = CopilotProvider(mock_handler=mock_handler)
        agent = AgentDef(
            name="test_agent",
            model="gpt-4",
            prompt="Test prompt",
        )

        tools = ["scrape_url", "file_read"]
        await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Test prompt",
            tools=tools,
        )

        # Verify via call history
        call_history = provider.get_call_history()
        assert call_history[0]["tools"] == ["scrape_url", "file_read"]

    @pytest.mark.asyncio
    async def test_multiple_agents_with_different_tools(self) -> None:
        """Test that multiple agent calls track tools independently."""
        provider = CopilotProvider()

        agent1 = AgentDef(name="agent1", model="gpt-4", prompt="Prompt 1")
        agent2 = AgentDef(name="agent2", model="gpt-4", prompt="Prompt 2")
        agent3 = AgentDef(name="agent3", model="gpt-4", prompt="Prompt 3")

        await provider.execute(agent1, {}, "Prompt 1", tools=["tool_a", "tool_b"])
        await provider.execute(agent2, {}, "Prompt 2", tools=[])
        await provider.execute(agent3, {}, "Prompt 3", tools=None)

        history = provider.get_call_history()
        assert len(history) == 3
        assert history[0]["agent_name"] == "agent1"
        assert history[0]["tools"] == ["tool_a", "tool_b"]
        assert history[1]["agent_name"] == "agent2"
        assert history[1]["tools"] == []
        assert history[2]["agent_name"] == "agent3"
        assert history[2]["tools"] is None
