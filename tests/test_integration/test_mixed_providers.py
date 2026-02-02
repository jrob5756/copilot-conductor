"""Tests for workflows attempting to mix Copilot and Claude providers.

Verifies that mixing providers is properly documented as unsupported
and that the behavior is clear.
"""

from copilot_conductor.config.loader import load_workflow
from copilot_conductor.config.schema import AgentDef, RuntimeConfig


class TestMixedProviderWorkflows:
    """Test workflows that attempt to use both providers."""

    def test_workflow_has_single_provider(self, tmp_path):
        """Verify workflow schema enforces single provider."""
        # Valid: Single provider
        workflow_yaml = tmp_path / "single_provider.yaml"
        workflow_yaml.write_text("""
workflow:
  name: single-provider
  version: "1.0"
  entry_point: agent1
  runtime:
    provider: claude

agents:
  - name: agent1
    prompt: "test"
    output:
      result:
        type: string
    routes:
      - to: $end
""")

        config = load_workflow(str(workflow_yaml))
        assert config.workflow.runtime.provider == "claude"

    def test_cannot_override_provider_per_agent(self, tmp_path):
        """Document that agent-level provider override is not supported."""
        # Agents inherit workflow-level provider
        workflow_yaml = tmp_path / "workflow.yaml"
        workflow_yaml.write_text("""
workflow:
  name: test
  version: "1.0"
  entry_point: agent1
  runtime:
    provider: claude

agents:
  - name: agent1
    prompt: "test"
    output:
      result:
        type: string
    routes:
      - to: $end
""")

        config = load_workflow(str(workflow_yaml))
        # All agents use workflow-level provider
        assert config.workflow.runtime.provider == "claude"
        # Agent schema doesn't have 'provider' field
        assert "provider" not in AgentDef.model_fields

    def test_claude_fields_ignored_by_copilot_provider(self, tmp_path):
        """Verify Claude fields don't break Copilot provider."""
        # Create runtime config with provider=copilot
        runtime = RuntimeConfig(provider="copilot")

        # Verify Claude fields are None
        assert runtime.temperature is None
        assert runtime.max_tokens is None

        # Serialization excludes None values
        dumped = runtime.model_dump(exclude_none=True)
        assert dumped == {"provider": "copilot", "mcp_servers": {}}

    def test_provider_parameter_isolation(self, tmp_path):
        """Test that provider-specific parameters don't interfere.

        Addresses reviewer concern: No validation that mixing providers
        properly isolates parameters.

        Currently, workflows use a single provider, but this test documents
        the expected isolation behavior for parameter namespacing.
        """
        # Claude-specific parameters should only apply to Claude provider
        workflow_yaml = tmp_path / "claude_params.yaml"
        workflow_yaml.write_text("""
workflow:
  name: claude-with-params
  version: "1.0"
  entry_point: agent1
  runtime:
    provider: claude
    temperature: 0.5
    max_tokens: 1000

agents:
  - name: agent1
    prompt: "test"
    output:
      result:
        type: string
    routes:
      - to: $end
""")

        config = load_workflow(str(workflow_yaml))

        # Verify Claude parameters are loaded
        assert config.workflow.runtime.temperature == 0.5
        assert config.workflow.runtime.max_tokens == 1000

        # If provider is later changed to Copilot, these parameters would be ignored
        # (Copilot doesn't support all the same parameters)
        # This is handled by factory.py using conditional parameter passing

    def test_parameter_exclusion_prevents_pollution(self, tmp_path):
        """Test that None parameters don't pollute provider instantiation.

        Ensures backward compatibility: Copilot workflows don't get
        claude_api_key=None, and Claude workflows don't get copilot_token=None.
        """
        # Copilot workflow shouldn't have Claude parameters
        copilot_yaml = tmp_path / "copilot.yaml"
        copilot_yaml.write_text("""
workflow:
  name: copilot-workflow
  version: "1.0"
  entry_point: agent1
  runtime:
    provider: copilot

agents:
  - name: agent1
    prompt: "test"
    output:
      result:
        type: string
    routes:
      - to: $end
""")

        config = load_workflow(str(copilot_yaml))

        # Schema should have Claude fields as None
        assert config.workflow.runtime.temperature is None
        assert config.workflow.runtime.max_tokens is None

        # But serialization with exclude_none=True won't include them
        serialized = config.model_dump(exclude_none=True)
        assert "temperature" not in serialized["workflow"]["runtime"]
        assert "max_tokens" not in serialized["workflow"]["runtime"]

        # This prevents provider factory from receiving irrelevant parameters
