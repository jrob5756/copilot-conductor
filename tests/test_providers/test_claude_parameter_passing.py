"""End-to-end parameter passing tests for Claude provider.

Verifies that all 6 Claude-specific parameters are properly passed through:
- temperature
- max_tokens
- top_p
- top_k
- stop_sequences
- metadata

Tests the full chain: factory -> provider -> SDK
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from copilot_conductor.config.schema import AgentDef
from copilot_conductor.providers.factory import create_provider


class TestClaudeParameterPassing:
    """Tests for end-to-end parameter passing through factory."""

    @patch("copilot_conductor.providers.factory.ClaudeProvider")
    def test_all_claude_parameters_passed_from_factory(
        self, mock_claude_class: Mock
    ) -> None:
        """Test that factory passes all Claude parameters to provider."""
        mock_instance = Mock()
        mock_claude_class.return_value = mock_instance

        # Create provider with all Claude parameters
        runtime_config = {
            "provider": "claude",
            "api_key": "test-key",
            "model": "claude-3-opus-20240229",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
            "top_k": 50,
            "stop_sequences": ["STOP", "END"],
            "metadata": {"user_id": "test_user"},
        }

        create_provider(runtime_config)

        # Verify all parameters were passed to ClaudeProvider constructor
        mock_claude_class.assert_called_once_with(
            api_key="test-key",
            model="claude-3-opus-20240229",
            temperature=0.7,
            max_tokens=4096,
            top_p=0.9,
            top_k=50,
            stop_sequences=["STOP", "END"],
            metadata={"user_id": "test_user"},
        )

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_all_claude_parameters_passed_to_sdk(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that provider passes all parameters to Claude SDK."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))

        # Mock successful response
        mock_response = Mock()
        mock_response.content = [
            Mock(type="text", text="Test response")
        ]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_class.return_value = mock_client

        # Import after patching
        from copilot_conductor.providers.claude import ClaudeProvider

        # Create provider with all parameters
        provider = ClaudeProvider(
            api_key="test-key",
            model="claude-3-opus-20240229",
            temperature=0.7,
            max_tokens=4096,
            top_p=0.9,
            top_k=50,
            stop_sequences=["STOP", "END"],
            metadata={"user_id": "test_user"},
        )

        # Execute agent
        agent = AgentDef(name="test_agent", prompt="Test prompt", model="claude-3-sonnet-20240229")
        context = {}
        rendered_prompt = "Test prompt"

        await provider.execute(agent, context, rendered_prompt)

        # Verify SDK was called with all parameters
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-sonnet-20240229"  # Agent model overrides
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["stop_sequences"] == ["STOP", "END"]
        assert call_kwargs["metadata"] == {"user_id": "test_user"}
        assert call_kwargs["messages"] == [{"role": "user", "content": "Test prompt"}]

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_optional_parameters_not_passed_when_none(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that optional parameters are not passed to SDK when None."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))

        # Mock successful response
        mock_response = Mock()
        mock_response.content = [
            Mock(type="text", text="Test response")
        ]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_class.return_value = mock_client

        # Import after patching
        from copilot_conductor.providers.claude import ClaudeProvider

        # Create provider with minimal parameters (all optional params are None)
        provider = ClaudeProvider()

        # Execute agent
        agent = AgentDef(name="test_agent", prompt="Test prompt")
        context = {}
        rendered_prompt = "Test prompt"

        await provider.execute(agent, context, rendered_prompt)

        # Verify SDK was called without optional parameters
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" not in call_kwargs  # None, so not passed
        assert "top_p" not in call_kwargs
        assert "top_k" not in call_kwargs
        assert "stop_sequences" not in call_kwargs
        assert "metadata" not in call_kwargs
        # Required parameters should still be present
        assert "model" in call_kwargs
        assert "max_tokens" in call_kwargs
        assert "messages" in call_kwargs

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_agent_model_overrides_provider_model(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that agent-level model overrides provider default."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))

        # Mock successful response
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Test response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_class.return_value = mock_client

        # Import after patching
        from copilot_conductor.providers.claude import ClaudeProvider

        # Create provider with default model
        provider = ClaudeProvider(model="claude-3-5-sonnet-latest")

        # Execute agent with different model
        agent = AgentDef(
            name="test_agent",
            prompt="Test prompt",
            model="claude-3-opus-20240229"
        )
        context = {}
        rendered_prompt = "Test prompt"

        await provider.execute(agent, context, rendered_prompt)

        # Verify agent model was used
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-opus-20240229"
