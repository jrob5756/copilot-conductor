"""Workflow execution engine for Copilot Conductor.

This module provides the WorkflowEngine class for orchestrating
multi-agent workflow execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from copilot_conductor.engine.context import WorkflowContext
from copilot_conductor.engine.router import Router, RouteResult
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
        self.router = Router()

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

            # Evaluate routes using the Router
            route_result = self._evaluate_routes(agent, output.content)

            if route_result.target == "$end":
                return self._build_final_output(route_result.output_transform)

            current_agent_name = route_result.target

    def _find_agent(self, name: str) -> AgentDef | None:
        """Find agent by name.

        Args:
            name: The agent name to find.

        Returns:
            The agent definition if found, None otherwise.
        """
        return next((a for a in self.config.agents if a.name == name), None)

    def _get_next_agent(self, agent: AgentDef, output: dict[str, Any]) -> str:
        """Get next agent from routes (legacy method, use _evaluate_routes instead).

        This method is kept for backward compatibility but delegates to _evaluate_routes.

        Args:
            agent: The current agent definition.
            output: The agent's output content.

        Returns:
            The name of the next agent or "$end".
        """
        result = self._evaluate_routes(agent, output)
        return result.target

    def _evaluate_routes(self, agent: AgentDef, output: dict[str, Any]) -> RouteResult:
        """Evaluate routes using the Router.

        Uses the Router to evaluate routing rules and determine the next agent.
        Supports both Jinja2 template conditions and simpleeval arithmetic expressions.

        Args:
            agent: The current agent definition.
            output: The agent's output content.

        Returns:
            RouteResult with target and optional output transform.
        """
        if not agent.routes:
            # No routes defined - default to $end
            return RouteResult(target="$end")

        # Build context for condition evaluation
        eval_context = self.context.get_for_template()

        return self.router.evaluate(agent.routes, output, eval_context)

    def _build_final_output(
        self, route_output_transform: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Build final output using output templates.

        Renders each output template expression with the full context.
        If a route output transform is provided, it will be merged with
        the template-rendered output (transform values take precedence).

        Args:
            route_output_transform: Optional output values from the $end route.

        Returns:
            Dict with rendered output values.
        """
        ctx = self.context.get_for_template()
        result: dict[str, Any] = {}

        for key, template in self.config.output.items():
            rendered = self.renderer.render(template, ctx)
            # Try to parse as JSON if it looks like JSON
            result[key] = self._maybe_parse_json(rendered)

        # Merge route output transform if provided (takes precedence)
        if route_output_transform:
            for key, value in route_output_transform.items():
                result[key] = self._maybe_parse_json(value) if isinstance(value, str) else value

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
