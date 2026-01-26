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
