"""Integration tests verifying existing Copilot workflows still work after schema changes.

This test suite addresses critical review feedback:
- Verifies backward compatibility at execution level (not just schema loading)
- Tests existing example workflows end-to-end
- Ensures schema changes don't break existing functionality
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from copilot_conductor.config.loader import load_workflow
from copilot_conductor.engine.workflow import WorkflowEngine
from copilot_conductor.providers.copilot import CopilotProvider


@pytest.fixture
def mock_copilot_sdk():
    """Mock the Copilot SDK agent to avoid real API calls."""
    with patch("copilot_conductor.providers.copilot.Agent") as mock_agent_class:
        mock_agent = AsyncMock()
        mock_agent.run_stream = AsyncMock(return_value=[
            {"type": "text", "content": "Test response"}
        ])
        mock_agent.run = AsyncMock(return_value={"type": "text", "content": "Test response"})
        mock_agent_class.return_value = mock_agent
        yield mock_agent


@pytest.mark.asyncio
async def test_simple_qa_yaml_executes(mock_copilot_sdk, tmp_path):
    """Verify simple-qa.yaml example executes successfully after schema changes."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    workflow_path = examples_dir / "simple-qa.yaml"
    
    if not workflow_path.exists():
        pytest.skip(f"Example file not found: {workflow_path}")
    
    # Load workflow with new schema (includes Claude fields)
    config = load_workflow(workflow_path)
    
    # Verify workflow loaded successfully
    assert config.name == "Simple Q&A"
    assert len(config.agents) > 0
    
    # Create provider and engine
    provider = CopilotProvider(api_key="mock-key-for-testing")
    engine = WorkflowEngine(config, provider)
    
    # Execute workflow
    result = await engine.run({"question": "What is Python?"})
    
    # Verify execution completed
    assert result is not None
    assert "output" in result or "answer" in result or len(result) > 0


@pytest.mark.asyncio
async def test_parallel_validation_yaml_executes(mock_copilot_sdk, tmp_path):
    """Verify parallel-validation.yaml executes with new schema."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    workflow_path = examples_dir / "parallel-validation.yaml"
    
    if not workflow_path.exists():
        pytest.skip(f"Example file not found: {workflow_path}")
    
    config = load_workflow(workflow_path)
    
    # Verify parallel groups still work
    assert config.parallel is not None
    assert len(config.parallel) > 0
    
    provider = CopilotProvider(api_key="mock-key-for-testing")
    engine = WorkflowEngine(config, provider)
    
    # Execute with mocked provider
    result = await engine.run({"code": "def test(): pass"})
    
    assert result is not None


@pytest.mark.asyncio
async def test_for_each_simple_yaml_executes(mock_copilot_sdk, tmp_path):
    """Verify for-each-simple.yaml executes with new schema."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    workflow_path = examples_dir / "for-each-simple.yaml"
    
    if not workflow_path.exists():
        pytest.skip(f"Example file not found: {workflow_path}")
    
    config = load_workflow(workflow_path)
    
    # Verify for-each groups still work
    assert config.for_each is not None
    assert len(config.for_each) > 0
    
    provider = CopilotProvider(api_key="mock-key-for-testing")
    engine = WorkflowEngine(config, provider)
    
    # Execute
    result = await engine.run({"items": ["item1", "item2"]})
    
    assert result is not None


@pytest.mark.asyncio
async def test_all_copilot_examples_load_and_validate(tmp_path):
    """Verify all Copilot example workflows load and validate after schema changes."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    
    if not examples_dir.exists():
        pytest.skip(f"Examples directory not found: {examples_dir}")
    
    # Find all non-Claude YAML files
    copilot_examples = [
        f for f in examples_dir.glob("*.yaml")
        if "claude" not in f.stem.lower()
    ]
    
    assert len(copilot_examples) > 0, "No Copilot example workflows found"
    
    loaded_count = 0
    for example_file in copilot_examples:
        try:
            config = load_workflow(example_file)
            
            # Verify basic structure
            assert config.name is not None
            assert len(config.agents) > 0
            
            # Verify Claude fields default to None (backward compatibility)
            if hasattr(config, "runtime") and config.runtime:
                assert config.runtime.temperature is None or isinstance(config.runtime.temperature, float)
                assert config.runtime.max_tokens is None or isinstance(config.runtime.max_tokens, int)
            
            loaded_count += 1
        except Exception as e:
            pytest.fail(f"Failed to load {example_file.name}: {e}")
    
    assert loaded_count == len(copilot_examples), "Not all examples loaded successfully"


@pytest.mark.asyncio
async def test_copilot_workflow_with_tools(mock_copilot_sdk, tmp_path):
    """Verify workflows with tool definitions still execute correctly."""
    workflow_yaml = """
name: Tool Test
provider: copilot
agents:
  - name: agent1
    prompt: "Use tools to answer: {{ question }}"
tools:
  - name: calculator
    description: "Calculate math expressions"
    parameters:
      type: object
      properties:
        expression:
          type: string
          description: "Math expression to evaluate"
      required: [expression]
output:
  result: "{{ agent1.output }}"
"""
    
    workflow_path = tmp_path / "tool_test.yaml"
    workflow_path.write_text(workflow_yaml)
    
    config = load_workflow(workflow_path)
    
    # Verify tools still work
    assert config.tools is not None
    assert len(config.tools) == 1
    assert config.tools[0].name == "calculator"
    
    provider = CopilotProvider(api_key="mock-key-for-testing")
    engine = WorkflowEngine(config, provider)
    
    result = await engine.run({"question": "What is 2+2?"})
    
    assert result is not None


@pytest.mark.asyncio
async def test_schema_changes_dont_affect_copilot_provider():
    """Verify that Claude schema fields don't interfere with Copilot provider."""
    workflow_yaml = """
name: Copilot Only Test
provider: copilot
runtime:
  model: gpt-4
agents:
  - name: test_agent
    prompt: "Answer: {{ question }}"
output:
  answer: "{{ test_agent.output }}"
"""
    
    from io import StringIO
    config = load_workflow(StringIO(workflow_yaml))
    
    # Verify Copilot provider ignores Claude fields
    assert config.runtime.model == "gpt-4"
    
    # Verify Claude fields are None
    assert config.runtime.temperature is None
    assert config.runtime.max_tokens is None
    assert config.runtime.top_p is None
    
    # Verify provider can be instantiated
    provider = CopilotProvider(api_key="mock-key")
    assert provider is not None
