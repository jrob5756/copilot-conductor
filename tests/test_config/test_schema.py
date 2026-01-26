"""Tests for the Pydantic schema models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from copilot_conductor.config.schema import (
    AgentDef,
    ContextConfig,
    GateOption,
    HooksConfig,
    InputDef,
    LimitsConfig,
    OutputField,
    RouteDef,
    RuntimeConfig,
    WorkflowConfig,
    WorkflowDef,
)


class TestInputDef:
    """Tests for InputDef model."""

    def test_valid_string_input(self) -> None:
        """Test creating a valid string input."""
        input_def = InputDef(type="string", required=True, description="A test input")
        assert input_def.type == "string"
        assert input_def.required is True
        assert input_def.default is None

    def test_valid_input_with_default(self) -> None:
        """Test creating an input with a default value."""
        input_def = InputDef(type="string", required=False, default="hello")
        assert input_def.default == "hello"

    def test_number_input_with_valid_default(self) -> None:
        """Test number input with valid numeric default."""
        input_def = InputDef(type="number", default=42)
        assert input_def.default == 42

    def test_number_input_with_float_default(self) -> None:
        """Test number input with float default."""
        input_def = InputDef(type="number", default=3.14)
        assert input_def.default == 3.14

    def test_boolean_input_with_valid_default(self) -> None:
        """Test boolean input with valid boolean default."""
        input_def = InputDef(type="boolean", default=True)
        assert input_def.default is True

    def test_array_input_with_valid_default(self) -> None:
        """Test array input with valid list default."""
        input_def = InputDef(type="array", default=["a", "b"])
        assert input_def.default == ["a", "b"]

    def test_object_input_with_valid_default(self) -> None:
        """Test object input with valid dict default."""
        input_def = InputDef(type="object", default={"key": "value"})
        assert input_def.default == {"key": "value"}

    def test_invalid_default_type_raises(self) -> None:
        """Test that mismatched default type raises ValidationError."""
        with pytest.raises(ValidationError):
            InputDef(type="string", default=123)

    def test_invalid_type_raises(self) -> None:
        """Test that invalid type raises ValidationError."""
        with pytest.raises(ValidationError):
            InputDef(type="invalid_type")  # type: ignore


class TestOutputField:
    """Tests for OutputField model."""

    def test_simple_string_output(self) -> None:
        """Test creating a simple string output field."""
        output = OutputField(type="string", description="A result")
        assert output.type == "string"
        assert output.description == "A result"

    def test_array_output_with_items(self) -> None:
        """Test array output with item schema."""
        output = OutputField(
            type="array",
            items=OutputField(type="string"),
        )
        assert output.type == "array"
        assert output.items is not None
        assert output.items.type == "string"

    def test_object_output_with_properties(self) -> None:
        """Test object output with properties."""
        output = OutputField(
            type="object",
            properties={
                "name": OutputField(type="string"),
                "count": OutputField(type="number"),
            },
        )
        assert output.type == "object"
        assert output.properties is not None
        assert "name" in output.properties
        assert output.properties["name"].type == "string"


class TestRouteDef:
    """Tests for RouteDef model."""

    def test_simple_route(self) -> None:
        """Test creating a simple unconditional route."""
        route = RouteDef(to="next_agent")
        assert route.to == "next_agent"
        assert route.when is None

    def test_conditional_route(self) -> None:
        """Test creating a conditional route."""
        route = RouteDef(to="success_agent", when="{{ output.success }}")
        assert route.to == "success_agent"
        assert route.when == "{{ output.success }}"

    def test_route_with_output_transform(self) -> None:
        """Test route with output transformation."""
        route = RouteDef(
            to="next",
            output={"result": "{{ output.value }}"},
        )
        assert route.output == {"result": "{{ output.value }}"}

    def test_end_route(self) -> None:
        """Test route to $end."""
        route = RouteDef(to="$end")
        assert route.to == "$end"

    def test_empty_target_raises(self) -> None:
        """Test that empty route target raises ValidationError."""
        with pytest.raises(ValidationError):
            RouteDef(to="")


class TestGateOption:
    """Tests for GateOption model."""

    def test_simple_option(self) -> None:
        """Test creating a simple gate option."""
        option = GateOption(label="Approve", value="approved", route="next_agent")
        assert option.label == "Approve"
        assert option.value == "approved"
        assert option.route == "next_agent"
        assert option.prompt_for is None

    def test_option_with_prompt_for(self) -> None:
        """Test gate option with text input prompt."""
        option = GateOption(
            label="Request Changes",
            value="changes",
            route="reviewer",
            prompt_for="feedback",
        )
        assert option.prompt_for == "feedback"


class TestContextConfig:
    """Tests for ContextConfig model."""

    def test_default_values(self) -> None:
        """Test default context configuration."""
        config = ContextConfig()
        assert config.mode == "accumulate"
        assert config.max_tokens is None
        assert config.trim_strategy is None

    def test_explicit_mode(self) -> None:
        """Test explicit context mode."""
        config = ContextConfig(mode="explicit")
        assert config.mode == "explicit"

    def test_with_trimming(self) -> None:
        """Test context config with trimming options."""
        config = ContextConfig(
            mode="accumulate",
            max_tokens=4000,
            trim_strategy="truncate",
        )
        assert config.max_tokens == 4000
        assert config.trim_strategy == "truncate"

    def test_invalid_mode_raises(self) -> None:
        """Test that invalid mode raises ValidationError."""
        with pytest.raises(ValidationError):
            ContextConfig(mode="invalid")  # type: ignore


class TestLimitsConfig:
    """Tests for LimitsConfig model."""

    def test_default_values(self) -> None:
        """Test default limits configuration."""
        config = LimitsConfig()
        assert config.max_iterations == 10
        assert config.timeout_seconds == 600

    def test_custom_limits(self) -> None:
        """Test custom limits."""
        config = LimitsConfig(max_iterations=50, timeout_seconds=1200)
        assert config.max_iterations == 50
        assert config.timeout_seconds == 1200

    def test_max_iterations_bounds(self) -> None:
        """Test max_iterations bounds validation."""
        with pytest.raises(ValidationError):
            LimitsConfig(max_iterations=0)  # Below minimum

        with pytest.raises(ValidationError):
            LimitsConfig(max_iterations=101)  # Above maximum

    def test_timeout_bounds(self) -> None:
        """Test timeout_seconds bounds validation."""
        with pytest.raises(ValidationError):
            LimitsConfig(timeout_seconds=0)  # Below minimum

        with pytest.raises(ValidationError):
            LimitsConfig(timeout_seconds=3601)  # Above maximum


class TestHooksConfig:
    """Tests for HooksConfig model."""

    def test_empty_hooks(self) -> None:
        """Test empty hooks configuration."""
        config = HooksConfig()
        assert config.on_start is None
        assert config.on_complete is None
        assert config.on_error is None

    def test_all_hooks(self) -> None:
        """Test all hooks configured."""
        config = HooksConfig(
            on_start="starting",
            on_complete="completed",
            on_error="error occurred",
        )
        assert config.on_start == "starting"
        assert config.on_complete == "completed"
        assert config.on_error == "error occurred"


class TestAgentDef:
    """Tests for AgentDef model."""

    def test_minimal_agent(self) -> None:
        """Test creating a minimal agent."""
        agent = AgentDef(name="agent1", model="gpt-4", prompt="Hello")
        assert agent.name == "agent1"
        assert agent.model == "gpt-4"
        assert agent.type is None
        assert agent.routes == []
        assert agent.input == []

    def test_agent_with_all_fields(self) -> None:
        """Test agent with all fields populated."""
        agent = AgentDef(
            name="full_agent",
            description="A fully configured agent",
            type="agent",
            model="gpt-4",
            input=["workflow.input.goal"],
            tools=["web_search"],
            system_prompt="You are helpful.",
            prompt="Process: {{ workflow.input.goal }}",
            output={"result": OutputField(type="string")},
            routes=[RouteDef(to="$end")],
        )
        assert agent.description == "A fully configured agent"
        assert agent.type == "agent"
        assert len(agent.tools) == 1
        assert agent.system_prompt == "You are helpful."

    def test_human_gate_with_options(self) -> None:
        """Test human_gate agent with options."""
        agent = AgentDef(
            name="gate1",
            type="human_gate",
            prompt="Choose an option:",
            options=[
                GateOption(label="Yes", value="yes", route="next"),
                GateOption(label="No", value="no", route="$end"),
            ],
        )
        assert agent.type == "human_gate"
        assert len(agent.options) == 2

    def test_human_gate_without_options_raises(self) -> None:
        """Test that human_gate without options raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentDef(name="gate1", type="human_gate", prompt="Choose:")
        assert "options" in str(exc_info.value)

    def test_human_gate_without_prompt_raises(self) -> None:
        """Test that human_gate without prompt raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentDef(
                name="gate1",
                type="human_gate",
                options=[GateOption(label="Ok", value="ok", route="next")],
            )
        assert "prompt" in str(exc_info.value)


class TestRuntimeConfig:
    """Tests for RuntimeConfig model."""

    def test_default_values(self) -> None:
        """Test default runtime configuration."""
        config = RuntimeConfig()
        assert config.provider == "copilot"
        assert config.default_model is None

    def test_custom_provider(self) -> None:
        """Test custom provider setting."""
        config = RuntimeConfig(provider="openai-agents", default_model="gpt-4")
        assert config.provider == "openai-agents"
        assert config.default_model == "gpt-4"

    def test_invalid_provider_raises(self) -> None:
        """Test that invalid provider raises ValidationError."""
        with pytest.raises(ValidationError):
            RuntimeConfig(provider="invalid")  # type: ignore


class TestWorkflowDef:
    """Tests for WorkflowDef model."""

    def test_minimal_workflow(self) -> None:
        """Test minimal workflow definition."""
        workflow = WorkflowDef(name="test", entry_point="agent1")
        assert workflow.name == "test"
        assert workflow.entry_point == "agent1"
        assert workflow.runtime.provider == "copilot"

    def test_full_workflow(self) -> None:
        """Test fully configured workflow definition."""
        workflow = WorkflowDef(
            name="full",
            description="A full workflow",
            version="1.0.0",
            entry_point="start",
            runtime=RuntimeConfig(provider="copilot"),
            input={"goal": InputDef(type="string")},
            context=ContextConfig(mode="explicit"),
            limits=LimitsConfig(max_iterations=20),
            hooks=HooksConfig(on_start="starting"),
        )
        assert workflow.version == "1.0.0"
        assert workflow.context.mode == "explicit"
        assert workflow.limits.max_iterations == 20


class TestWorkflowConfig:
    """Tests for WorkflowConfig model."""

    def test_minimal_config(self) -> None:
        """Test minimal valid workflow configuration."""
        config = WorkflowConfig(
            workflow=WorkflowDef(name="test", entry_point="agent1"),
            agents=[
                AgentDef(name="agent1", model="gpt-4", prompt="Hello", routes=[RouteDef(to="$end")])
            ],
        )
        assert config.workflow.name == "test"
        assert len(config.agents) == 1

    def test_entry_point_validation(self) -> None:
        """Test that entry_point must exist in agents."""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowConfig(
                workflow=WorkflowDef(name="test", entry_point="nonexistent"),
                agents=[AgentDef(name="agent1", model="gpt-4", prompt="Hello")],
            )
        assert "entry_point" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_route_target_validation(self) -> None:
        """Test that route targets must exist."""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowConfig(
                workflow=WorkflowDef(name="test", entry_point="agent1"),
                agents=[
                    AgentDef(
                        name="agent1",
                        model="gpt-4",
                        prompt="Hello",
                        routes=[RouteDef(to="unknown_agent")],
                    )
                ],
            )
        assert "unknown_agent" in str(exc_info.value)

    def test_end_route_is_valid(self) -> None:
        """Test that $end is always a valid route target."""
        config = WorkflowConfig(
            workflow=WorkflowDef(name="test", entry_point="agent1"),
            agents=[
                AgentDef(name="agent1", model="gpt-4", prompt="Hello", routes=[RouteDef(to="$end")])
            ],
        )
        assert config.agents[0].routes[0].to == "$end"

    def test_multi_agent_routing(self) -> None:
        """Test routing between multiple agents."""
        config = WorkflowConfig(
            workflow=WorkflowDef(name="test", entry_point="agent1"),
            agents=[
                AgentDef(
                    name="agent1",
                    model="gpt-4",
                    prompt="Step 1",
                    routes=[RouteDef(to="agent2")],
                ),
                AgentDef(
                    name="agent2",
                    model="gpt-4",
                    prompt="Step 2",
                    routes=[RouteDef(to="$end")],
                ),
            ],
        )
        assert len(config.agents) == 2
        assert config.agents[0].routes[0].to == "agent2"

    def test_workflow_with_tools(self) -> None:
        """Test workflow with tools configuration."""
        config = WorkflowConfig(
            workflow=WorkflowDef(name="test", entry_point="agent1"),
            tools=["web_search", "calculator"],
            agents=[
                AgentDef(
                    name="agent1",
                    model="gpt-4",
                    prompt="Hello",
                    tools=["web_search"],
                    routes=[RouteDef(to="$end")],
                )
            ],
        )
        assert len(config.tools) == 2
        assert config.agents[0].tools == ["web_search"]

    def test_workflow_with_output(self) -> None:
        """Test workflow with output templates."""
        config = WorkflowConfig(
            workflow=WorkflowDef(name="test", entry_point="agent1"),
            agents=[
                AgentDef(name="agent1", model="gpt-4", prompt="Hello", routes=[RouteDef(to="$end")])
            ],
            output={
                "result": "{{ agent1.output }}",
                "summary": "Completed",
            },
        )
        assert len(config.output) == 2
