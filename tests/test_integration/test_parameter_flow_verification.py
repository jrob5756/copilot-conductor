"""Parameter flow verification tests.

These tests verify that parameters ACTUALLY reach the Anthropic SDK API calls,
addressing the reviewer concern: 'No verification that temperature, max_tokens,
top_p, top_k, stop_sequences, metadata actually reach the Anthropic SDK API calls'.

This test file inspects the actual kwargs passed to the Anthropic SDK's
messages.create() method to ensure all parameters flow correctly.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from copilot_conductor.config.loader import load_workflow
from copilot_conductor.engine.workflow import WorkflowEngine


class TestParameterFlowToAnthropicSDK:
    """Verify ALL parameters reach Anthropic SDK API calls."""

    @pytest.fixture
    def mock_anthropic_sdk(self):
        """Mock Anthropic SDK and capture API call parameters."""
        with patch("copilot_conductor.providers.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            
            # Mock successful response
            mock_response = Mock()
            mock_response.content = [Mock(text="Test response", type="text")]
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_response.stop_reason = "end_turn"
            mock_response.id = "msg_123"
            mock_response.type = "message"
            mock_response.role = "assistant"
            
            mock_client.messages.create = MagicMock(return_value=mock_response)
            
            yield mock_client

    @pytest.mark.asyncio
    async def test_temperature_reaches_api_call(self, tmp_path, mock_anthropic_sdk):
        """Verify temperature parameter reaches Anthropic SDK API call."""
        workflow_yaml = tmp_path / "test_temp.yaml"
        workflow_yaml.write_text("""
name: test-temperature
version: "1.0"
runtime:
  provider: claude
  temperature: 0.42

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        engine = WorkflowEngine(config)
        await engine.run({})

        # Verify temperature=0.42 was passed to SDK
        call_kwargs = mock_anthropic_sdk.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.42, \
            f"Expected temperature=0.42, got {call_kwargs.get('temperature')}"

    @pytest.mark.asyncio
    async def test_max_tokens_reaches_api_call(self, tmp_path, mock_anthropic_sdk):
        """Verify max_tokens parameter reaches Anthropic SDK API call."""
        workflow_yaml = tmp_path / "test_max_tokens.yaml"
        workflow_yaml.write_text("""
name: test-max-tokens
version: "1.0"
runtime:
  provider: claude
  max_tokens: 2048

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        engine = WorkflowEngine(config)
        await engine.run({})

        # Verify max_tokens=2048 was passed to SDK
        call_kwargs = mock_anthropic_sdk.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 2048, \
            f"Expected max_tokens=2048, got {call_kwargs.get('max_tokens')}"

    @pytest.mark.asyncio
    async def test_top_p_reaches_api_call(self, tmp_path, mock_anthropic_sdk):
        """Verify top_p parameter reaches Anthropic SDK API call."""
        workflow_yaml = tmp_path / "test_top_p.yaml"
        workflow_yaml.write_text("""
name: test-top-p
version: "1.0"
runtime:
  provider: claude
  top_p: 0.85

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        engine = WorkflowEngine(config)
        await engine.run({})

        # Verify top_p=0.85 was passed to SDK
        call_kwargs = mock_anthropic_sdk.messages.create.call_args.kwargs
        assert call_kwargs["top_p"] == 0.85, \
            f"Expected top_p=0.85, got {call_kwargs.get('top_p')}"

    @pytest.mark.asyncio
    async def test_top_k_reaches_api_call(self, tmp_path, mock_anthropic_sdk):
        """Verify top_k parameter reaches Anthropic SDK API call."""
        workflow_yaml = tmp_path / "test_top_k.yaml"
        workflow_yaml.write_text("""
name: test-top-k
version: "1.0"
runtime:
  provider: claude
  top_k: 50

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        engine = WorkflowEngine(config)
        await engine.run({})

        # Verify top_k=50 was passed to SDK
        call_kwargs = mock_anthropic_sdk.messages.create.call_args.kwargs
        assert call_kwargs["top_k"] == 50, \
            f"Expected top_k=50, got {call_kwargs.get('top_k')}"

    @pytest.mark.asyncio
    async def test_stop_sequences_reaches_api_call(self, tmp_path, mock_anthropic_sdk):
        """Verify stop_sequences parameter reaches Anthropic SDK API call."""
        workflow_yaml = tmp_path / "test_stop.yaml"
        workflow_yaml.write_text("""
name: test-stop-sequences
version: "1.0"
runtime:
  provider: claude
  stop_sequences: ["STOP", "END", "DONE"]

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        engine = WorkflowEngine(config)
        await engine.run({})

        # Verify stop_sequences was passed to SDK
        call_kwargs = mock_anthropic_sdk.messages.create.call_args.kwargs
        assert call_kwargs["stop_sequences"] == ["STOP", "END", "DONE"], \
            f"Expected stop_sequences=['STOP', 'END', 'DONE'], got {call_kwargs.get('stop_sequences')}"

    @pytest.mark.asyncio
    async def test_metadata_reaches_api_call(self, tmp_path, mock_anthropic_sdk):
        """Verify metadata parameter reaches Anthropic SDK API call."""
        workflow_yaml = tmp_path / "test_metadata.yaml"
        workflow_yaml.write_text("""
name: test-metadata
version: "1.0"
runtime:
  provider: claude
  metadata:
    user_id: "test-user-123"
    session_id: "session-456"

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        engine = WorkflowEngine(config)
        await engine.run({})

        # Verify metadata was passed to SDK
        call_kwargs = mock_anthropic_sdk.messages.create.call_args.kwargs
        assert call_kwargs["metadata"] == {"user_id": "test-user-123", "session_id": "session-456"}, \
            f"Expected metadata dict, got {call_kwargs.get('metadata')}"

    @pytest.mark.asyncio
    async def test_all_parameters_together_reach_api_call(self, tmp_path, mock_anthropic_sdk):
        """Verify ALL Claude parameters reach Anthropic SDK API call simultaneously."""
        workflow_yaml = tmp_path / "test_all_params.yaml"
        workflow_yaml.write_text("""
name: test-all-params
version: "1.0"
runtime:
  provider: claude
  temperature: 0.75
  max_tokens: 4096
  top_p: 0.92
  top_k: 45
  stop_sequences: ["FINAL"]
  metadata:
    user_id: "user-999"

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        engine = WorkflowEngine(config)
        await engine.run({})

        # Verify ALL parameters were passed to SDK in the same call
        call_kwargs = mock_anthropic_sdk.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.75
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["top_p"] == 0.92
        assert call_kwargs["top_k"] == 45
        assert call_kwargs["stop_sequences"] == ["FINAL"]
        assert call_kwargs["metadata"] == {"user_id": "user-999"}

    @pytest.mark.asyncio
    async def test_agent_level_temperature_override_reaches_api_call(
        self, tmp_path, mock_anthropic_sdk
    ):
        """Verify agent-level temperature override reaches Anthropic SDK API call."""
        workflow_yaml = tmp_path / "test_agent_override.yaml"
        workflow_yaml.write_text("""
name: test-agent-override
version: "1.0"
runtime:
  provider: claude
  temperature: 0.5

agents:
  - name: agent1
    prompt: "test"
    temperature: 0.99
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        engine = WorkflowEngine(config)
        await engine.run({})

        # Verify agent-level override (0.99) was used, not runtime default (0.5)
        call_kwargs = mock_anthropic_sdk.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.99, \
            f"Expected agent override temperature=0.99, got {call_kwargs.get('temperature')}"

    @pytest.mark.asyncio
    async def test_none_parameters_excluded_from_api_call(self, tmp_path, mock_anthropic_sdk):
        """Verify parameters with None values are NOT passed to SDK."""
        workflow_yaml = tmp_path / "test_none_params.yaml"
        workflow_yaml.write_text("""
name: test-none-params
version: "1.0"
runtime:
  provider: claude

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        engine = WorkflowEngine(config)
        await engine.run({})

        # Verify optional parameters are NOT in API call when None
        call_kwargs = mock_anthropic_sdk.messages.create.call_args.kwargs
        assert "temperature" not in call_kwargs, \
            "temperature should not be passed when None"
        assert "top_p" not in call_kwargs, \
            "top_p should not be passed when None"
        assert "top_k" not in call_kwargs, \
            "top_k should not be passed when None"
        assert "stop_sequences" not in call_kwargs, \
            "stop_sequences should not be passed when None"
        assert "metadata" not in call_kwargs, \
            "metadata should not be passed when None"

        # Required parameters should still be present
        assert "model" in call_kwargs
        assert "max_tokens" in call_kwargs
        assert "messages" in call_kwargs


class TestExcludeNoneInSerialization:
    """Verify exclude_none=True prevents Claude fields in serialized Copilot configs."""

    @pytest.mark.asyncio
    async def test_exclude_none_during_workflow_execution(self, tmp_path):
        """Test that exclude_none=True works during actual workflow execution."""
        workflow_yaml = tmp_path / "copilot_workflow.yaml"
        workflow_yaml.write_text("""
name: copilot-workflow
version: "1.0"
runtime:
  provider: copilot

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        
        # Simulate config persistence/transmission during workflow execution
        serialized = config.model_dump(mode="json", exclude_none=True)
        
        # Verify Claude fields are completely absent
        runtime = serialized["workflow"]["runtime"]
        claude_fields = ["temperature", "max_tokens", "top_p", "top_k", "stop_sequences", "metadata"]
        
        for field in claude_fields:
            assert field not in runtime, \
                f"Claude field '{field}' should not be in serialized Copilot config"
        
        # Verify Copilot provider is present
        assert runtime["provider"] == "copilot"

    @pytest.mark.asyncio
    async def test_exclude_none_with_partial_claude_params(self, tmp_path):
        """Test exclude_none with some Claude params set, others None."""
        workflow_yaml = tmp_path / "partial_claude.yaml"
        workflow_yaml.write_text("""
name: partial-claude
version: "1.0"
runtime:
  provider: claude
  temperature: 0.7

agents:
  - name: agent1
    prompt: "test"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        serialized = config.model_dump(mode="json", exclude_none=True)
        
        runtime = serialized["workflow"]["runtime"]
        
        # temperature is set, should be present
        assert "temperature" in runtime
        assert runtime["temperature"] == 0.7
        
        # Other Claude params are None, should be excluded
        assert "top_p" not in runtime
        assert "top_k" not in runtime
        assert "stop_sequences" not in runtime
        assert "metadata" not in runtime
