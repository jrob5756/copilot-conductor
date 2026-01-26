"""Workflow context management for Copilot Conductor.

This module provides the WorkflowContext class for managing workflow state,
including inputs, agent outputs, and execution history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowContext:
    """Manages workflow execution state and context accumulation.

    The WorkflowContext stores all state needed during workflow execution:
    - workflow_inputs: Initial inputs provided when starting the workflow
    - agent_outputs: Outputs from each executed agent
    - current_iteration: Counter for iteration limit enforcement
    - execution_history: Ordered list of executed agent names

    Context modes:
    - accumulate: All prior agent outputs are available to subsequent agents
    - last_only: Only the most recent agent's output is available
    - explicit: Only inputs explicitly declared in the agent's input list are available

    Example:
        >>> ctx = WorkflowContext()
        >>> ctx.set_workflow_inputs({"question": "What is Python?"})
        >>> ctx.store("answerer", {"answer": "A programming language"})
        >>> agent_ctx = ctx.build_for_agent("checker", ["answerer.output"], "accumulate")
        >>> agent_ctx["answerer"]["output"]["answer"]
        'A programming language'
    """

    workflow_inputs: dict[str, Any] = field(default_factory=dict)
    """Inputs provided at workflow start."""

    agent_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Outputs from executed agents, keyed by agent name."""

    current_iteration: int = 0
    """Current execution iteration count."""

    execution_history: list[str] = field(default_factory=list)
    """Ordered list of executed agent names."""

    def set_workflow_inputs(self, inputs: dict[str, Any]) -> None:
        """Store workflow-level inputs.

        Args:
            inputs: Dictionary of input values provided at workflow start.
        """
        self.workflow_inputs = inputs.copy()

    def store(self, agent_name: str, output: dict[str, Any]) -> None:
        """Store an agent's output in context.

        This method updates the agent_outputs dictionary, appends the agent
        to execution_history, and increments the iteration counter.

        Args:
            agent_name: The name of the agent whose output is being stored.
            output: The structured output from the agent.
        """
        self.agent_outputs[agent_name] = output
        self.execution_history.append(agent_name)
        self.current_iteration += 1

    def build_for_agent(
        self,
        agent_name: str,
        inputs: list[str],
        mode: str = "accumulate",
    ) -> dict[str, Any]:
        """Build context dict for a specific agent based on its input declarations.

        The context includes:
        - workflow: Contains workflow-level inputs under workflow.input
        - context: Metadata about execution (iteration, history)
        - Agent outputs: Based on the accumulation mode

        Args:
            agent_name: Name of the agent needing context.
            inputs: List of input references (e.g., ['workflow.input.goal', 'planner.output']).
            mode: Context mode - accumulate, last_only, or explicit.

        Returns:
            Dict with 'workflow', agent outputs, and 'context' metadata.

        Raises:
            KeyError: If explicit mode is used and a required (non-optional) input is missing.
        """
        # For explicit mode, start with empty workflow inputs
        # For other modes, include all workflow inputs
        if mode == "explicit":
            ctx: dict[str, Any] = {
                "workflow": {"input": {}},
                "context": {
                    "iteration": self.current_iteration,
                    "history": self.execution_history.copy(),
                },
            }
            # Only declared inputs - parse input references
            for input_ref in inputs:
                self._add_explicit_input(ctx, input_ref)
        else:
            ctx = {
                "workflow": {"input": self.workflow_inputs.copy()},
                "context": {
                    "iteration": self.current_iteration,
                    "history": self.execution_history.copy(),
                },
            }

            if mode == "accumulate":
                # All prior agent outputs available
                for agent, output in self.agent_outputs.items():
                    ctx[agent] = {"output": output}

            elif mode == "last_only" and self.execution_history:
                # Only the most recent agent's output
                last_agent = self.execution_history[-1]
                ctx[last_agent] = {"output": self.agent_outputs.get(last_agent, {})}

        return ctx

    def _add_explicit_input(self, ctx: dict[str, Any], input_ref: str) -> None:
        """Add an explicit input reference to the context.

        Handles optional dependencies with '?' suffix - missing optional
        dependencies are silently skipped instead of raising errors.

        Input reference formats:
        - workflow.input.param_name - References a workflow input
        - agent_name.output - References an agent's entire output
        - agent_name.output.field - References a specific output field
        - agent_name.field - Shorthand for agent_name.output.field (deprecated but supported)
        - Any reference with '?' suffix - Optional dependency

        Args:
            ctx: The context dictionary to update.
            input_ref: The input reference string.

        Raises:
            KeyError: If a required (non-optional) input is not found.
        """
        # Check for optional suffix
        is_optional = input_ref.endswith("?")
        ref = input_ref.rstrip("?")

        parts = ref.split(".")

        if len(parts) < 2:
            if not is_optional:
                raise KeyError(f"Invalid input reference format: {input_ref}")
            return

        if parts[0] == "workflow":
            # workflow.input.param_name format
            if len(parts) >= 3 and parts[1] == "input":
                param_name = parts[2]
                if param_name in self.workflow_inputs:
                    # Ensure workflow.input exists in ctx
                    if "workflow" not in ctx:
                        ctx["workflow"] = {"input": {}}
                    elif "input" not in ctx["workflow"]:
                        ctx["workflow"]["input"] = {}
                    ctx["workflow"]["input"][param_name] = self.workflow_inputs[param_name]
                elif not is_optional:
                    raise KeyError(f"Missing required workflow input: {param_name}")
        else:
            # agent_name.output or agent_name.output.field or agent_name.field format
            agent_name = parts[0]

            if agent_name in self.agent_outputs:
                # Ensure the agent context exists
                if agent_name not in ctx:
                    ctx[agent_name] = {"output": {}}
                elif "output" not in ctx[agent_name]:
                    ctx[agent_name]["output"] = {}

                # If just agent_name.output, copy entire output
                if len(parts) == 2 and parts[1] == "output":
                    ctx[agent_name]["output"] = self.agent_outputs[agent_name].copy()
                elif len(parts) >= 3 and parts[1] == "output":
                    # Copy specific field(s): agent_name.output.field
                    field_name = parts[2]
                    if field_name in self.agent_outputs[agent_name]:
                        ctx[agent_name]["output"][field_name] = self.agent_outputs[
                            agent_name
                        ][field_name]
                    elif not is_optional:
                        raise KeyError(
                            f"Missing output field '{field_name}' from agent '{agent_name}'"
                        )
                elif len(parts) == 2 and parts[1] != "output":
                    # Shorthand format: agent_name.field -> agent_name.output.field
                    field_name = parts[1]
                    if field_name in self.agent_outputs[agent_name]:
                        ctx[agent_name]["output"][field_name] = self.agent_outputs[
                            agent_name
                        ][field_name]
                    elif not is_optional:
                        raise KeyError(
                            f"Missing output field '{field_name}' from agent '{agent_name}'"
                        )
            elif not is_optional:
                raise KeyError(f"Missing required agent output: {agent_name}")

    def get_for_template(self) -> dict[str, Any]:
        """Get full context for template rendering.

        Returns a context dictionary with all agent outputs and workflow
        inputs available for use in output template expressions.

        Returns:
            Dict with workflow inputs and all agent outputs.
        """
        return self.build_for_agent("__template__", [], mode="accumulate")

    def get_latest_output(self) -> dict[str, Any] | None:
        """Get the output from the most recently executed agent.

        Returns:
            The output dictionary from the last agent, or None if no agents executed.
        """
        if not self.execution_history:
            return None
        last_agent = self.execution_history[-1]
        return self.agent_outputs.get(last_agent)
