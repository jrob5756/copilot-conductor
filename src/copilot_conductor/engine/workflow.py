"""Workflow execution engine for Copilot Conductor.

This module provides the WorkflowEngine class for orchestrating
multi-agent workflow execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from copilot_conductor.engine.context import WorkflowContext
from copilot_conductor.exceptions import ExecutionError
from copilot_conductor.executor.agent import AgentExecutor
from copilot_conductor.executor.template import TemplateRenderer

if TYPE_CHECKING:
    from copilot_conductor.config.schema import AgentDef, WorkflowConfig
    from copilot_conductor.providers.base import AgentProvider


class WorkflowEngine:
    """Orchestrates multi-agent workflow execution.

    The WorkflowEngine manages the complete lifecycle of a workflow:
    1. Initialize context with workflow inputs
    2. Execute agents in sequence following routing rules
    3. Accumulate context between agents
    4. Build final output from templates

    Example:
        >>> from copilot_conductor.config.loader import load_workflow
        >>> from copilot_conductor.providers.factory import create_provider
        >>> config = load_workflow("workflow.yaml")
        >>> provider = await create_provider(config.workflow.runtime.provider)
        >>> engine = WorkflowEngine(config, provider)
        >>> result = await engine.run({"question": "What is Python?"})
    """

    def __init__(
        self,
        config: WorkflowConfig,
        provider: AgentProvider,
    ) -> None:
        """Initialize the WorkflowEngine.

        Args:
            config: The workflow configuration.
            provider: The agent provider for execution.
        """
        self.config = config
        self.provider = provider
        self.context = WorkflowContext()
        self.executor = AgentExecutor(provider)
        self.renderer = TemplateRenderer()

    async def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the workflow from entry_point to $end.

        This is the main entry point for workflow execution. It:
        1. Sets up the context with the provided inputs
        2. Executes agents in sequence based on routing rules
        3. Returns the final output built from output templates

        Args:
            inputs: Workflow input values.

        Returns:
            Final output dict built from output templates.

        Raises:
            ExecutionError: If an agent is not found or execution fails.
            ValidationError: If agent output doesn't match schema.
            TemplateError: If template rendering fails.
        """
        self.context.set_workflow_inputs(inputs)
        current_agent_name = self.config.workflow.entry_point

        while True:
            agent = self._find_agent(current_agent_name)
            if agent is None:
                raise ExecutionError(
                    f"Agent not found: {current_agent_name}",
                    suggestion=f"Ensure '{current_agent_name}' is defined in the workflow",
                )

            # Skip human gates for now (EPIC-009)
            if agent.type == "human_gate":
                # Placeholder - will be implemented in EPIC-009
                # For now, just follow the first route
                if agent.routes:
                    next_agent = agent.routes[0].to
                elif agent.options:
                    next_agent = agent.options[0].route
                else:
                    next_agent = "$end"

                if next_agent == "$end":
                    return self._build_final_output()
                current_agent_name = next_agent
                continue

            # Build context for this agent
            agent_context = self.context.build_for_agent(
                agent.name,
                agent.input,
                mode=self.config.workflow.context.mode,
            )

            # Execute agent
            output = await self.executor.execute(agent, agent_context)

            # Store output
            self.context.store(agent.name, output.content)

            # Evaluate routes (basic for now - full routing in EPIC-006)
            next_agent = self._get_next_agent(agent, output.content)

            if next_agent == "$end":
                return self._build_final_output()

            current_agent_name = next_agent

    def _find_agent(self, name: str) -> AgentDef | None:
        """Find agent by name.

        Args:
            name: The agent name to find.

        Returns:
            The agent definition if found, None otherwise.
        """
        return next((a for a in self.config.agents if a.name == name), None)

    def _get_next_agent(self, agent: AgentDef, output: dict[str, Any]) -> str:
        """Get next agent from routes.

        This is a basic implementation that evaluates routes in order.
        Full conditional routing with expression evaluation will be
        implemented in EPIC-006.

        Args:
            agent: The current agent definition.
            output: The agent's output content.

        Returns:
            The name of the next agent or "$end".
        """
        if not agent.routes:
            return "$end"

        # Build context for condition evaluation
        eval_context = self.context.get_for_template()
        eval_context["output"] = output

        # Evaluate routes in order (first match wins)
        for route in agent.routes:
            if route.when is None:
                # Unconditional route - always matches
                return route.to

            # Evaluate the condition using template renderer
            try:
                condition_result = self.renderer.evaluate_condition(route.when, eval_context)
                if condition_result:
                    return route.to
            except Exception:
                # If condition evaluation fails, skip this route
                continue

        # No matching route - default to $end
        return "$end"

    def _build_final_output(self) -> dict[str, Any]:
        """Build final output using output templates.

        Renders each output template expression with the full context.

        Returns:
            Dict with rendered output values.
        """
        ctx = self.context.get_for_template()
        result: dict[str, Any] = {}

        for key, template in self.config.output.items():
            rendered = self.renderer.render(template, ctx)
            # Try to parse as JSON if it looks like JSON
            result[key] = self._maybe_parse_json(rendered)

        return result

    @staticmethod
    def _maybe_parse_json(value: str) -> Any:
        """Attempt to parse a string as JSON.

        Args:
            value: The string to parse.

        Returns:
            Parsed JSON value if successful, original string otherwise.
        """
        import json

        stripped = value.strip()
        if stripped.startswith(("{", "[", '"')) or stripped in ("true", "false", "null"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        # Try to convert numeric strings
        try:
            if "." in stripped:
                return float(stripped)
            return int(stripped)
        except ValueError:
            pass
        return value

    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of the workflow execution.

        Returns:
            Dict with execution statistics.
        """
        return {
            "iterations": self.context.current_iteration,
            "agents_executed": self.context.execution_history.copy(),
            "context_mode": self.config.workflow.context.mode,
        }
