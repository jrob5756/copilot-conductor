"""Integration tests verifying schema fields are correctly passed to providers.

This module tests that all Claude-specific schema fields (temperature, max_tokens,
top_p, top_k, stop_sequences, metadata) are correctly passed from the schema
to the ClaudeProvider constructor and used during execution.

These tests use real provider classes (not mocks) to verify actual integration.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from copilot_conductor.config.schema import (
    AgentDef,
    OutputField,
    RouteDef,
    RuntimeConfig,
    WorkflowConfig,
    WorkflowDef,
)
from copilot_conductor.engine.workflow import WorkflowEngine
from copilot_conductor.providers.base import AgentOutput
from copilot_conductor.providers.claude import ClaudeProvider
from copilot_conductor.providers.factory import ProviderFactory


class TestSchemaToProviderIntegration:
    """Test that schema fields correctly integrate with provider implementations."""

    @pytest.mark.asyncio
    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    async def test_claude_runtime_config_fields_passed_to_provider(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ):
        """Test that all Claude runtime config fields are passed to ClaudeProvider.
        
        Verifies: temperature, max_tokens, top_p, top_k, stop_sequences, metadata
        """
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        
        # Mock the messages.create method
        mock_message = Mock()
        mock_message.id = "msg_123"
        mock_message.type = "message"
        mock_message.role = "assistant"
        mock_message.model = "claude-3-5-sonnet-latest"
        mock_message.stop_reason = "end_turn"
        mock_message.usage = Mock(input_tokens=10, output_tokens=20, cache_creation_input_tokens=0)
        mock_message.content = [Mock(type="text", text='{"answer": "test"}')]
        
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        mock_anthropic_class.return_value = mock_client
        
        # Create workflow config with all Claude fields
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="test-claude-fields",
                description="Test Claude fields",
                version="1.0.0",
                entry_point="agent1",
                runtime=RuntimeConfig(
                    provider="claude",
                    temperature=0.8,
                    max_tokens=2048,
                    top_p=0.9,
                    top_k=40,
                    stop_sequences=["STOP", "END"],
                    metadata={"user_id": "test123"}
                )
            ),
            agents=[
                AgentDef(
                    name="agent1",
                    description="Test agent",
                    model="claude-3-5-sonnet-latest",
                    prompt="Test prompt",
                    output={"answer": OutputField(type="string")},
                    routes=[RouteDef(to="$end")]
                )
            ]
        )
        
        # Create provider using factory (real instantiation)
        provider = ProviderFactory.create_provider(config.workflow.runtime)
        
        # Verify provider is ClaudeProvider
        assert isinstance(provider, ClaudeProvider)
        
        # Verify all fields were set correctly on the provider
        assert provider._default_temperature == 0.8
        assert provider._default_max_tokens == 2048
        assert provider._default_top_p == 0.9
        assert provider._default_top_k == 40
        assert provider._default_stop_sequences == ["STOP", "END"]
        assert provider._default_metadata == {"user_id": "test123"}
        
        # Execute through engine to verify fields are used
        engine = WorkflowEngine(config, provider)
        result = await engine.run({})
        
        # Verify messages.create was called with correct parameters
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["stop_sequences"] == ["STOP", "END"]
        assert call_kwargs["metadata"] == {"user_id": "test123"}
        
        await engine.cleanup()

    @pytest.mark.asyncio
    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    async def test_claude_provider_with_none_fields(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ):
        """Test that ClaudeProvider handles None values for optional fields correctly."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        
        # Mock the messages.create method
        mock_message = Mock()
        mock_message.id = "msg_123"
        mock_message.type = "message"
        mock_message.role = "assistant"
        mock_message.model = "claude-3-5-sonnet-latest"
        mock_message.stop_reason = "end_turn"
        mock_message.usage = Mock(input_tokens=10, output_tokens=20, cache_creation_input_tokens=0)
        mock_message.content = [Mock(type="text", text='{"result": "ok"}')]
        
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        mock_anthropic_class.return_value = mock_client
        
        # Create workflow config with all Claude fields set to None
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="test-none-fields",
                description="Test None fields",
                version="1.0.0",
                entry_point="agent1",
                runtime=RuntimeConfig(
                    provider="claude",
                    temperature=None,
                    max_tokens=None,
                    top_p=None,
                    top_k=None,
                    stop_sequences=None,
                    metadata=None
                )
            ),
            agents=[
                AgentDef(
                    name="agent1",
                    description="Test agent",
                    model="claude-3-5-sonnet-latest",
                    prompt="Test prompt",
                    output={"result": OutputField(type="string")},
                    routes=[RouteDef(to="$end")]
                )
            ]
        )
        
        # Create provider using factory
        provider = ProviderFactory.create_provider(config.workflow.runtime)
        
        # Verify provider defaults are used when config fields are None
        assert isinstance(provider, ClaudeProvider)
        assert provider._default_temperature == 1.0  # Default from ClaudeProvider
        assert provider._default_max_tokens == 8192  # Default from ClaudeProvider
        
        # Execute workflow
        engine = WorkflowEngine(config, provider)
        result = await engine.run({})
        
        # Verify messages.create was called with defaults (None fields excluded)
        call_kwargs = mock_client.messages.create.call_args.kwargs
        # When fields are None, they should use provider defaults
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["max_tokens"] == 8192
        assert "top_p" not in call_kwargs  # None values excluded
        assert "top_k" not in call_kwargs
        assert "stop_sequences" not in call_kwargs
        assert "metadata" not in call_kwargs
        
        await engine.cleanup()

    @pytest.mark.asyncio
    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    async def test_agent_level_overrides_runtime_defaults(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ):
        """Test that agent-level config overrides runtime defaults."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        
        # Mock the messages.create method
        mock_message = Mock()
        mock_message.id = "msg_123"
        mock_message.type = "message"
        mock_message.role = "assistant"
        mock_message.model = "claude-3-5-sonnet-latest"
        mock_message.stop_reason = "end_turn"
        mock_message.usage = Mock(input_tokens=10, output_tokens=20, cache_creation_input_tokens=0)
        mock_message.content = [Mock(type="text", text='{"result": "ok"}')]
        
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        mock_anthropic_class.return_value = mock_client
        
        # Create workflow with runtime defaults
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="test-overrides",
                description="Test agent overrides",
                version="1.0.0",
                entry_point="agent1",
                runtime=RuntimeConfig(
                    provider="claude",
                    temperature=0.5,
                    max_tokens=1024
                )
            ),
            agents=[
                AgentDef(
                    name="agent1",
                    description="Test agent",
                    model="claude-3-5-sonnet-latest",
                    prompt="Test prompt",
                    temperature=0.9,  # Override runtime default
                    max_tokens=4096,  # Override runtime default
                    output={"result": OutputField(type="string")},
                    routes=[RouteDef(to="$end")]
                )
            ]
        )
        
        # Create provider
        provider = ProviderFactory.create_provider(config.workflow.runtime)
        
        # Execute workflow
        engine = WorkflowEngine(config, provider)
        result = await engine.run({})
        
        # Verify agent-level overrides were used
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9  # Agent override, not 0.5
        assert call_kwargs["max_tokens"] == 4096  # Agent override, not 1024
        
        await engine.cleanup()
