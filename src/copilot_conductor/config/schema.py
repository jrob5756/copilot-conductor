"""Pydantic models for workflow configuration.

This module defines all Pydantic models for validating and parsing
workflow YAML configuration files.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class InputDef(BaseModel):
    """Definition for a workflow input parameter."""

    type: Literal["string", "number", "boolean", "array", "object"]
    """The type of the input parameter."""

    required: bool = True
    """Whether the input is required."""

    default: Any = None
    """Default value if the input is not provided."""

    description: str | None = None
    """Human-readable description of the input."""

    @field_validator("default")
    @classmethod
    def validate_default_type(cls, v: Any, info) -> Any:
        """Ensure default value matches declared type."""
        if v is None:
            return v

        # Get the declared type from the data being validated
        type_value = info.data.get("type")
        if type_value is None:
            return v

        # Type validation based on declared type
        type_checks = {
            "string": lambda x: isinstance(x, str),
            "number": lambda x: isinstance(x, int | float) and not isinstance(x, bool),
            "boolean": lambda x: isinstance(x, bool),
            "array": lambda x: isinstance(x, list),
            "object": lambda x: isinstance(x, dict),
        }

        check = type_checks.get(type_value)
        if check and not check(v):
            raise ValueError(
                f"default value must be of type '{type_value}', got {type(v).__name__}"
            )

        return v


class OutputField(BaseModel):
    """Schema for a single output field from an agent."""

    type: Literal["string", "number", "boolean", "array", "object"]
    """The type of the output field."""

    description: str | None = None
    """Human-readable description of the output field."""

    items: OutputField | None = None
    """For array types, the schema of array items."""

    properties: dict[str, OutputField] | None = None
    """For object types, the schema of object properties."""

    @model_validator(mode="after")
    def validate_type_specific_fields(self) -> OutputField:
        """Ensure type-specific fields are properly set."""
        if self.type == "array" and self.items is None:
            # Items are optional but recommended for arrays
            pass
        if self.type == "object" and self.properties is None:
            # Properties are optional but recommended for objects
            pass
        return self


class RouteDef(BaseModel):
    """Definition for a routing rule."""

    to: str
    """Target agent name, '$end', or human gate name."""

    when: str | None = None
    """Optional condition expression (Jinja2 template that evaluates to bool)."""

    output: dict[str, str] | None = None
    """Optional output transformation (template expressions)."""

    @field_validator("to")
    @classmethod
    def validate_target(cls, v: str) -> str:
        """Validate route target format."""
        if not v:
            raise ValueError("Route target cannot be empty")
        return v


class GateOption(BaseModel):
    """Option presented in a human gate."""

    label: str
    """Display text for the option."""

    value: str
    """Value stored when option selected."""

    route: str
    """Agent to route to when selected."""

    prompt_for: str | None = None
    """Optional: field name to prompt for text input."""


class ContextConfig(BaseModel):
    """Configuration for context accumulation behavior."""

    mode: Literal["accumulate", "last_only", "explicit"] = "accumulate"
    """
    Context accumulation mode:
    - accumulate: All prior outputs available (default)
    - last_only: Only previous agent's output available
    - explicit: Only inputs listed in the agent's `input` array are available;
                nothing is automatically accumulated from prior agents
    """

    max_tokens: int | None = None
    """Maximum context tokens before trimming."""

    trim_strategy: Literal["summarize", "truncate", "drop_oldest"] | None = None
    """Strategy for reducing context size when limit exceeded."""


class LimitsConfig(BaseModel):
    """Safety limits for workflow execution."""

    max_iterations: int = Field(default=10, ge=1, le=100)
    """Maximum number of agent executions before forced termination."""

    timeout_seconds: int = Field(default=600, ge=1, le=3600)
    """Maximum wall-clock time for entire workflow."""


class HooksConfig(BaseModel):
    """Lifecycle hooks for workflow events."""

    on_start: str | None = None
    """Expression evaluated when workflow starts."""

    on_complete: str | None = None
    """Expression evaluated when workflow completes successfully."""

    on_error: str | None = None
    """Expression evaluated when workflow fails."""


class AgentDef(BaseModel):
    """Definition for a single agent in the workflow."""

    name: str
    """Unique identifier for this agent."""

    description: str | None = None
    """Human-readable description of agent's purpose."""

    type: Literal["agent", "human_gate"] | None = None
    """Agent type. Defaults to 'agent' if not specified."""

    model: str | None = None
    """Model identifier (e.g., 'claude-sonnet-4'). Supports ${ENV:-default}."""

    input: list[str] = Field(default_factory=list)
    """Context dependencies. Format: 'agent_name.output' or 'workflow.input.param'.
    Suffix with '?' for optional dependencies."""

    tools: list[str] | None = None
    """Tools available to this agent. None = all, [] = none."""

    system_prompt: str | None = None
    """System message for the agent (always included)."""

    prompt: str = ""
    """User prompt template (Jinja2)."""

    output: dict[str, OutputField] | None = None
    """Expected output schema for validation."""

    routes: list[RouteDef] = Field(default_factory=list)
    """Routing rules evaluated in order after execution."""

    options: list[GateOption] | None = None
    """Options for human_gate type agents."""

    @model_validator(mode="after")
    def validate_agent_type(self) -> AgentDef:
        """Ensure agent has required fields for its type."""
        if self.type == "human_gate":
            if not self.options:
                raise ValueError("human_gate agents require 'options'")
            if not self.prompt:
                raise ValueError("human_gate agents require 'prompt'")
        return self


class RuntimeConfig(BaseModel):
    """Provider and runtime configuration."""

    provider: Literal["copilot", "openai-agents", "claude"] = "copilot"
    """SDK provider to use for agent execution."""

    default_model: str | None = None
    """Default model for agents that don't specify one."""


class WorkflowDef(BaseModel):
    """Top-level workflow configuration."""

    name: str
    """Unique workflow identifier."""

    description: str | None = None
    """Human-readable workflow description."""

    version: str | None = None
    """Semantic version string."""

    entry_point: str
    """Name of the first agent to execute."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    """Provider and runtime settings."""

    input: dict[str, InputDef] = Field(default_factory=dict)
    """Workflow input parameter definitions."""

    context: ContextConfig = Field(default_factory=ContextConfig)
    """Context accumulation settings."""

    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    """Execution safety limits."""

    hooks: HooksConfig | None = None
    """Lifecycle event hooks."""


class WorkflowConfig(BaseModel):
    """Complete workflow configuration file."""

    workflow: WorkflowDef
    """Workflow-level settings."""

    tools: list[str] = Field(default_factory=list)
    """Tools available to agents in this workflow."""

    agents: list[AgentDef]
    """Agent definitions."""

    output: dict[str, str] = Field(default_factory=dict)
    """Final output template expressions."""

    @model_validator(mode="after")
    def validate_references(self) -> WorkflowConfig:
        """Validate all agent references exist."""
        agent_names = {a.name for a in self.agents}

        # Validate entry_point
        if self.workflow.entry_point not in agent_names:
            raise ValueError(
                f"entry_point '{self.workflow.entry_point}' not found in agents"
            )

        # Validate route targets
        for agent in self.agents:
            for route in agent.routes:
                if route.to != "$end" and route.to not in agent_names:
                    raise ValueError(
                        f"Agent '{agent.name}' routes to unknown agent '{route.to}'"
                    )

        return self
