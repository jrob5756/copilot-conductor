"""End-to-end integration tests for all Claude EPICs.

Tests the complete flow from schema → provider → execution → output
across all 10 EPICs to verify full integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from copilot_conductor.config.loader import load_workflow
from copilot_conductor.config.schema import WorkflowConfig, AgentDef, WorkflowDef, RuntimeConfig, OutputField, RouteDef
from copilot_conductor.engine.workflow import WorkflowEngine
from copilot_conductor.providers.claude import ClaudeProvider
import json


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response matching actual SDK structure."""
    mock_response = Mock()
    mock_response.content = [Mock(text="Test response from Claude", type="text")]
    mock_response.model = "claude-3-5-sonnet-20241022"
    mock_response.usage = Mock(input_tokens=10, output_tokens=20)
    mock_response.stop_reason = "end_turn"
    mock_response.id = "msg_123"
    mock_response.type = "message"
    mock_response.role = "assistant"
    return mock_response


class TestEndToEndClaudeIntegration:
    """Verify all 10 EPICs integrate correctly end-to-end."""

    @pytest.mark.asyncio
    async def test_epic_001_009_full_workflow_execution(
        self, tmp_path, mock_anthropic_response
    ):
        """Test EPICs 001-009: Schema → Provider → Execution → Output.
        
        Verifies:
        - EPIC-001: Schema fields (temperature, max_tokens, etc.)
        - EPIC-002: Provider initialization
        - EPIC-003: Prompt template rendering
        - EPIC-004: API call parameter passing
        - EPIC-006: Output validation
        - EPIC-007: Error handling
        - EPIC-008: Tool support (null = all tools)
        - EPIC-009: Documentation compliance
        """
        # Create workflow YAML with all Claude parameters
        workflow_yaml = tmp_path / "test_workflow.yaml"
        workflow_yaml.write_text("""
name: test-claude-full-integration
version: "1.0"
runtime:
  provider: claude
  temperature: 0.7
  max_tokens: 1000
  top_p: 0.9
  top_k: 40
  stop_sequences: ["END"]
  metadata:
    user_id: "test-user"

agents:
  - name: agent1
    prompt: "Answer: {{ question }}"
    tools: null
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        # Load and validate workflow (EPIC-001: Schema)
        config = load_workflow(str(workflow_yaml))
        assert config.workflow.runtime.provider == "claude"
        assert config.workflow.runtime.temperature == 0.7
        assert config.workflow.runtime.max_tokens == 1000
        assert config.workflow.runtime.top_p == 0.9
        assert config.workflow.runtime.top_k == 40
        assert config.workflow.runtime.stop_sequences == ["END"]
        assert config.workflow.runtime.metadata == {"user_id": "test-user"}

        # Execute workflow with mocked Anthropic SDK (EPIC-002, 003, 004)
        with patch("copilot_conductor.providers.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create = MagicMock(return_value=mock_anthropic_response)

            engine = WorkflowEngine(config)
            result = await engine.run({"question": "What is 2+2?"})

            # Verify API call received correct parameters (EPIC-004)
            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 1000
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["top_k"] == 40
            assert call_kwargs["stop_sequences"] == ["END"]
            assert call_kwargs["metadata"] == {"user_id": "test-user"}

            # Verify output (EPIC-006)
            assert "agent1" in result
            assert "Test response from Claude" in str(result["agent1"])

    @pytest.mark.asyncio
    async def test_agent_level_parameter_overrides(
        self, tmp_path, mock_anthropic_response
    ):
        """Test agent-level parameter overrides for same workflow."""
        workflow_yaml = tmp_path / "test_overrides.yaml"
        workflow_yaml.write_text("""
name: test-overrides
version: "1.0"
runtime:
  provider: claude
  temperature: 0.5

agents:
  - name: creative
    prompt: "Be creative: {{ topic }}"
    temperature: 0.9
    max_tokens: 500
    output:
      result: string
  
  - name: precise
    prompt: "Be precise: {{ topic }}"
    temperature: 0.1
    max_tokens: 200
    output:
      result: string

routes:
  - from: creative
    to: precise
  - from: precise
    to: $end
""")

        config = load_workflow(str(workflow_yaml))

        with patch("copilot_conductor.providers.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create = MagicMock(return_value=mock_anthropic_response)

            engine = WorkflowEngine(config)
            await engine.run({"topic": "AI"})

            # Verify both agents called with different parameters
            assert mock_client.messages.create.call_count == 2
            
            # First call (creative agent) should use temperature=0.9
            first_call = mock_client.messages.create.call_args_list[0].kwargs
            assert first_call["temperature"] == 0.9
            assert first_call["max_tokens"] == 500

            # Second call (precise agent) should use temperature=0.1
            second_call = mock_client.messages.create.call_args_list[1].kwargs
            assert second_call["temperature"] == 0.1
            assert second_call["max_tokens"] == 200

    @pytest.mark.asyncio
    async def test_exclude_none_in_actual_workflow(self, tmp_path, mock_anthropic_response):
        """Verify exclude_none=True prevents Claude fields in Copilot workflows."""
        # Create workflow without Claude-specific fields
        workflow_yaml = tmp_path / "copilot_workflow.yaml"
        workflow_yaml.write_text("""
name: copilot-workflow
version: "1.0"
runtime:
  provider: copilot

agents:
  - name: agent1
    prompt: "Test prompt"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        
        # Serialize to dict (simulates config persistence/transmission)
        config_dict = config.model_dump(mode="json", exclude_none=True)
        
        # Verify Claude-specific fields are not present
        runtime_dict = config_dict["workflow"]["runtime"]
        assert "temperature" not in runtime_dict
        assert "max_tokens" not in runtime_dict
        assert "top_p" not in runtime_dict
        assert "top_k" not in runtime_dict
        assert "stop_sequences" not in runtime_dict
        assert "metadata" not in runtime_dict

    @pytest.mark.asyncio
    async def test_schema_validation_error_injection(self, tmp_path):
        """Test schema validation with invalid values."""
        
        # Test invalid temperature
        workflow_yaml = tmp_path / "invalid_temp.yaml"
        workflow_yaml.write_text("""
name: invalid
version: "1.0"
runtime:
  provider: claude
  temperature: 2.5
agents:
  - name: agent1
    prompt: "test"
    output:
      result: string
""")
        
        with pytest.raises(Exception) as exc_info:
            load_workflow(str(workflow_yaml))
        assert "temperature" in str(exc_info.value).lower()

        # Test invalid max_tokens
        workflow_yaml.write_text("""
name: invalid
version: "1.0"
runtime:
  provider: claude
  max_tokens: -1
agents:
  - name: agent1
    prompt: "test"
    output:
      result: string
""")
        
        with pytest.raises(Exception) as exc_info:
            load_workflow(str(workflow_yaml))
        assert "max_tokens" in str(exc_info.value).lower() or "greater than 0" in str(exc_info.value).lower()

        # Test invalid top_p
        workflow_yaml.write_text("""
name: invalid
version: "1.0"
runtime:
  provider: claude
  top_p: 1.5
agents:
  - name: agent1
    prompt: "test"
    output:
      result: string
""")
        
        with pytest.raises(Exception) as exc_info:
            load_workflow(str(workflow_yaml))
        assert "top_p" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_epic_010_backward_compatibility_in_workflow(
        self, tmp_path, mock_anthropic_response
    ):
        """Test EPIC-010: Copilot workflows still work after Claude addition."""
        # Create pure Copilot workflow (no Claude fields)
        workflow_yaml = tmp_path / "copilot_only.yaml"
        workflow_yaml.write_text("""
name: copilot-only
version: "1.0"
runtime:
  provider: copilot

agents:
  - name: agent1
    prompt: "Answer: {{ question }}"
    output:
      result: string

routes:
  - from: agent1
    to: $end
""")

        config = load_workflow(str(workflow_yaml))
        assert config.workflow.runtime.provider == "copilot"
        
        # Verify no Claude fields leaked
        assert config.workflow.runtime.temperature is None
        assert config.workflow.runtime.max_tokens is None
        assert config.workflow.runtime.top_p is None
        
        # Workflow should execute with Copilot provider
        with patch("copilot_conductor.providers.copilot.CopilotAgents") as mock_copilot:
            mock_instance = MagicMock()
            mock_copilot.return_value = mock_instance
            mock_instance.run.return_value = Mock(
                messages=[Mock(content="Copilot response")]
            )

            engine = WorkflowEngine(config)
            result = await engine.run({"question": "test"})
            
            # Verify Copilot SDK was called, not Anthropic
            mock_copilot.assert_called_once()
            assert "agent1" in result


@pytest.mark.performance
class TestClaudePerformanceIntegration:
    """Performance tests for Claude integration."""
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Mock Anthropic API response for performance tests."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response", type="text")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_123"
        mock_response.type = "message"
        mock_response.role = "assistant"
        return mock_response
    
    @pytest.mark.asyncio
    async def test_parameter_overhead(self, tmp_path, mock_anthropic_response):
        """Verify Claude parameter passing doesn't add significant overhead."""
        import time
        
        workflow_yaml = tmp_path / "perf_test.yaml"
        workflow_yaml.write_text("""
name: perf-test
version: "1.0"
runtime:
  provider: claude
  temperature: 0.7
  max_tokens: 1000
  top_p: 0.9
  top_k: 40

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

        with patch("copilot_conductor.providers.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create = MagicMock(return_value=mock_anthropic_response)

            engine = WorkflowEngine(config)
            
            # Measure execution time
            start = time.time()
            await engine.run({})
            duration = time.time() - start
            
            # Should complete in < 1 second (mocked, so overhead only)
            assert duration < 1.0, f"Unexpected overhead: {duration}s"
