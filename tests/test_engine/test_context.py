"""Unit tests for WorkflowContext.

Tests cover:
- Setting workflow inputs
- Storing agent outputs
- Building context for agents in different modes
- Optional dependencies with ? suffix
- Template context generation
"""

import pytest

from copilot_conductor.engine.context import WorkflowContext


class TestWorkflowContextBasic:
    """Basic WorkflowContext functionality tests."""

    def test_init_default_values(self) -> None:
        """Test WorkflowContext initializes with correct defaults."""
        ctx = WorkflowContext()

        assert ctx.workflow_inputs == {}
        assert ctx.agent_outputs == {}
        assert ctx.current_iteration == 0
        assert ctx.execution_history == []

    def test_set_workflow_inputs(self) -> None:
        """Test setting workflow inputs."""
        ctx = WorkflowContext()
        inputs = {"question": "What is Python?", "max_length": 100}

        ctx.set_workflow_inputs(inputs)

        assert ctx.workflow_inputs == inputs
        # Verify it's a copy, not the same reference
        inputs["new_key"] = "value"
        assert "new_key" not in ctx.workflow_inputs

    def test_store_agent_output(self) -> None:
        """Test storing agent output."""
        ctx = WorkflowContext()
        output = {"answer": "Python is a programming language"}

        ctx.store("answerer", output)

        assert ctx.agent_outputs["answerer"] == output
        assert ctx.execution_history == ["answerer"]
        assert ctx.current_iteration == 1

    def test_store_multiple_agents(self) -> None:
        """Test storing outputs from multiple agents."""
        ctx = WorkflowContext()

        ctx.store("agent1", {"result": "first"})
        ctx.store("agent2", {"result": "second"})
        ctx.store("agent3", {"result": "third"})

        assert ctx.current_iteration == 3
        assert ctx.execution_history == ["agent1", "agent2", "agent3"]
        assert ctx.agent_outputs["agent1"]["result"] == "first"
        assert ctx.agent_outputs["agent2"]["result"] == "second"
        assert ctx.agent_outputs["agent3"]["result"] == "third"

    def test_get_latest_output(self) -> None:
        """Test getting the latest output."""
        ctx = WorkflowContext()

        # No outputs yet
        assert ctx.get_latest_output() is None

        ctx.store("agent1", {"result": "first"})
        assert ctx.get_latest_output() == {"result": "first"}

        ctx.store("agent2", {"result": "second"})
        assert ctx.get_latest_output() == {"result": "second"}


class TestWorkflowContextAccumulateMode:
    """Tests for accumulate context mode."""

    def test_accumulate_mode_includes_all_outputs(self) -> None:
        """Test that accumulate mode includes all prior outputs."""
        ctx = WorkflowContext()
        ctx.set_workflow_inputs({"goal": "test"})
        ctx.store("planner", {"plan": "step 1"})
        ctx.store("executor", {"result": "done"})

        agent_ctx = ctx.build_for_agent("reviewer", [], mode="accumulate")

        # Should have workflow inputs
        assert agent_ctx["workflow"]["input"]["goal"] == "test"

        # Should have all agent outputs
        assert agent_ctx["planner"]["output"]["plan"] == "step 1"
        assert agent_ctx["executor"]["output"]["result"] == "done"

        # Should have context metadata
        assert agent_ctx["context"]["iteration"] == 2
        assert agent_ctx["context"]["history"] == ["planner", "executor"]

    def test_accumulate_mode_empty_outputs(self) -> None:
        """Test accumulate mode with no prior outputs."""
        ctx = WorkflowContext()
        ctx.set_workflow_inputs({"input": "value"})

        agent_ctx = ctx.build_for_agent("first_agent", [], mode="accumulate")

        assert agent_ctx["workflow"]["input"]["input"] == "value"
        assert agent_ctx["context"]["iteration"] == 0
        assert agent_ctx["context"]["history"] == []


class TestWorkflowContextLastOnlyMode:
    """Tests for last_only context mode."""

    def test_last_only_mode_includes_only_last_output(self) -> None:
        """Test that last_only mode only includes the most recent output."""
        ctx = WorkflowContext()
        ctx.set_workflow_inputs({"goal": "test"})
        ctx.store("planner", {"plan": "step 1"})
        ctx.store("executor", {"result": "done"})

        agent_ctx = ctx.build_for_agent("reviewer", [], mode="last_only")

        # Should have workflow inputs
        assert agent_ctx["workflow"]["input"]["goal"] == "test"

        # Should only have the last agent's output
        assert "planner" not in agent_ctx
        assert agent_ctx["executor"]["output"]["result"] == "done"

        # Should have context metadata
        assert agent_ctx["context"]["iteration"] == 2

    def test_last_only_mode_empty_history(self) -> None:
        """Test last_only mode with no prior agents."""
        ctx = WorkflowContext()
        ctx.set_workflow_inputs({"input": "value"})

        agent_ctx = ctx.build_for_agent("first_agent", [], mode="last_only")

        # Only workflow and context should be present
        assert "workflow" in agent_ctx
        assert "context" in agent_ctx


class TestWorkflowContextExplicitMode:
    """Tests for explicit context mode."""

    def test_explicit_mode_workflow_input(self) -> None:
        """Test explicit mode with workflow input reference."""
        ctx = WorkflowContext()
        ctx.set_workflow_inputs({"question": "What?", "other": "ignored"})

        agent_ctx = ctx.build_for_agent(
            "agent",
            ["workflow.input.question"],
            mode="explicit",
        )

        assert agent_ctx["workflow"]["input"]["question"] == "What?"
        assert "other" not in agent_ctx["workflow"]["input"]

    def test_explicit_mode_agent_output(self) -> None:
        """Test explicit mode with agent output reference."""
        ctx = WorkflowContext()
        ctx.store("answerer", {"answer": "42", "confidence": 0.9})

        agent_ctx = ctx.build_for_agent(
            "checker",
            ["answerer.output"],
            mode="explicit",
        )

        assert agent_ctx["answerer"]["output"]["answer"] == "42"
        assert agent_ctx["answerer"]["output"]["confidence"] == 0.9

    def test_explicit_mode_specific_field(self) -> None:
        """Test explicit mode with specific field reference."""
        ctx = WorkflowContext()
        ctx.store("answerer", {"answer": "42", "confidence": 0.9})

        agent_ctx = ctx.build_for_agent(
            "checker",
            ["answerer.output.answer"],
            mode="explicit",
        )

        assert agent_ctx["answerer"]["output"]["answer"] == "42"
        assert "confidence" not in agent_ctx["answerer"]["output"]

    def test_explicit_mode_missing_required_raises(self) -> None:
        """Test that missing required input raises KeyError."""
        ctx = WorkflowContext()

        with pytest.raises(KeyError, match="Missing required agent output"):
            ctx.build_for_agent(
                "checker",
                ["missing_agent.output"],
                mode="explicit",
            )

    def test_explicit_mode_missing_workflow_input_raises(self) -> None:
        """Test that missing required workflow input raises KeyError."""
        ctx = WorkflowContext()
        ctx.set_workflow_inputs({})

        with pytest.raises(KeyError, match="Missing required workflow input"):
            ctx.build_for_agent(
                "agent",
                ["workflow.input.missing"],
                mode="explicit",
            )


class TestWorkflowContextOptionalDeps:
    """Tests for optional dependencies with ? suffix."""

    def test_optional_missing_agent_skipped(self) -> None:
        """Test that missing optional agent output is skipped."""
        ctx = WorkflowContext()

        # Should not raise
        agent_ctx = ctx.build_for_agent(
            "checker",
            ["optional_agent.output?"],
            mode="explicit",
        )

        assert "optional_agent" not in agent_ctx

    def test_optional_missing_workflow_input_skipped(self) -> None:
        """Test that missing optional workflow input is skipped."""
        ctx = WorkflowContext()
        ctx.set_workflow_inputs({})

        # Should not raise
        agent_ctx = ctx.build_for_agent(
            "agent",
            ["workflow.input.optional?"],
            mode="explicit",
        )

        # workflow.input should be empty or not contain the optional field
        assert "optional" not in agent_ctx.get("workflow", {}).get("input", {})

    def test_optional_present_is_included(self) -> None:
        """Test that present optional dependencies are included."""
        ctx = WorkflowContext()
        ctx.store("reviewer", {"feedback": "looks good"})

        agent_ctx = ctx.build_for_agent(
            "executor",
            ["reviewer.feedback?"],
            mode="explicit",
        )

        assert agent_ctx["reviewer"]["output"]["feedback"] == "looks good"

    def test_optional_missing_field_skipped(self) -> None:
        """Test that missing optional field is skipped."""
        ctx = WorkflowContext()
        ctx.store("answerer", {"answer": "42"})

        # Should not raise even though 'confidence' doesn't exist
        agent_ctx = ctx.build_for_agent(
            "checker",
            ["answerer.output.confidence?"],
            mode="explicit",
        )

        # answerer should be in context but without confidence
        if "answerer" in agent_ctx:
            assert "confidence" not in agent_ctx["answerer"]["output"]

    def test_mixed_required_and_optional(self) -> None:
        """Test mixing required and optional dependencies."""
        ctx = WorkflowContext()
        ctx.set_workflow_inputs({"question": "What?"})
        ctx.store("answerer", {"answer": "42"})

        agent_ctx = ctx.build_for_agent(
            "summarizer",
            [
                "workflow.input.question",
                "answerer.output.answer",
                "missing_agent.output?",  # Optional, should be skipped
            ],
            mode="explicit",
        )

        assert agent_ctx["workflow"]["input"]["question"] == "What?"
        assert agent_ctx["answerer"]["output"]["answer"] == "42"
        assert "missing_agent" not in agent_ctx


class TestWorkflowContextGetForTemplate:
    """Tests for get_for_template method."""

    def test_get_for_template_includes_all(self) -> None:
        """Test that get_for_template returns full context."""
        ctx = WorkflowContext()
        ctx.set_workflow_inputs({"goal": "test"})
        ctx.store("agent1", {"output1": "value1"})
        ctx.store("agent2", {"output2": "value2"})

        template_ctx = ctx.get_for_template()

        assert template_ctx["workflow"]["input"]["goal"] == "test"
        assert template_ctx["agent1"]["output"]["output1"] == "value1"
        assert template_ctx["agent2"]["output"]["output2"] == "value2"
        assert template_ctx["context"]["iteration"] == 2

    def test_get_for_template_empty_context(self) -> None:
        """Test get_for_template with empty context."""
        ctx = WorkflowContext()

        template_ctx = ctx.get_for_template()

        assert template_ctx["workflow"]["input"] == {}
        assert template_ctx["context"]["iteration"] == 0
        assert template_ctx["context"]["history"] == []
