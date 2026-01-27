"""Workflow execution engine for Copilot Conductor.

This module provides the WorkflowEngine class for orchestrating
multi-agent workflow execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from copilot_conductor.engine.context import WorkflowContext
from copilot_conductor.engine.limits import LimitEnforcer
from copilot_conductor.engine.router import Router, RouteResult
from copilot_conductor.exceptions import ExecutionError
from copilot_conductor.executor.agent import AgentExecutor
from copilot_conductor.executor.template import TemplateRenderer
from copilot_conductor.gates.human import GateResult, HumanGateHandler

if TYPE_CHECKING:
    from copilot_conductor.config.schema import AgentDef, WorkflowConfig
    from copilot_conductor.providers.base import AgentProvider


@dataclass
class ExecutionStep:
    """A single step in the execution plan.

    Represents an agent that will be executed during workflow execution,
    along with its configuration and possible routing destinations.
    """

    agent_name: str
    """Name of the agent."""

    agent_type: str
    """Type: 'agent' or 'human_gate'."""

    model: str | None
    """Model used by this agent."""

    routes: list[dict[str, Any]] = field(default_factory=list)
    """Possible routes from this agent."""

    is_loop_target: bool = False
    """True if this agent could be a loop-back target."""


@dataclass
class ExecutionPlan:
    """Represents the workflow execution plan without actually running.

    This provides a static analysis of the workflow structure, showing
    all possible agents that may be executed and their routing paths.
    Used by the --dry-run flag to display the execution plan.
    """

    workflow_name: str
    """Name of the workflow."""

    entry_point: str
    """Name of the first agent."""

    steps: list[ExecutionStep] = field(default_factory=list)
    """Ordered steps in the execution plan."""

    max_iterations: int = 10
    """Maximum iterations configured."""

    timeout_seconds: int = 600
    """Timeout configured."""

    possible_paths: list[list[str]] = field(default_factory=list)
    """Possible execution paths through the workflow."""


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
        skip_gates: bool = False,
    ) -> None:
        """Initialize the WorkflowEngine.

        Args:
            config: The workflow configuration.
            provider: The agent provider for execution.
            skip_gates: If True, auto-selects first option at human gates.
        """
        self.config = config
        self.provider = provider
        self.skip_gates = skip_gates
        self.context = WorkflowContext()
        self.executor = AgentExecutor(provider, workflow_tools=config.tools)
        self.renderer = TemplateRenderer()
        self.router = Router()
        self.limits = LimitEnforcer(
            max_iterations=config.workflow.limits.max_iterations,
            timeout_seconds=config.workflow.limits.timeout_seconds,
        )
        self.gate_handler = HumanGateHandler(skip_gates=skip_gates)

    async def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the workflow from entry_point to $end.

        This is the main entry point for workflow execution. It:
        1. Sets up the context with the provided inputs
        2. Enforces iteration and timeout limits
        3. Executes agents in sequence based on routing rules
        4. Returns the final output built from output templates

        Args:
            inputs: Workflow input values.

        Returns:
            Final output dict built from output templates.

        Raises:
            ExecutionError: If an agent is not found or execution fails.
            MaxIterationsError: If max iterations limit is exceeded.
            TimeoutError: If timeout limit is exceeded.
            ValidationError: If agent output doesn't match schema.
            TemplateError: If template rendering fails.
        """
        self.context.set_workflow_inputs(inputs)
        self.limits.start()
        current_agent_name = self.config.workflow.entry_point

        async with self.limits.timeout_context():
            while True:
                agent = self._find_agent(current_agent_name)
                if agent is None:
                    raise ExecutionError(
                        f"Agent not found: {current_agent_name}",
                        suggestion=f"Ensure '{current_agent_name}' is defined in the workflow",
                    )

                # Check iteration limit before executing
                self.limits.check_iteration(current_agent_name)

                # Handle human gates
                if agent.type == "human_gate":
                    # Build context for the gate prompt
                    agent_context = self.context.get_for_template()

                    # Use the gate handler for interaction
                    gate_result: GateResult = await self.gate_handler.handle_gate(
                        agent, agent_context
                    )

                    # Store gate result in context
                    self.context.store(agent.name, {
                        "selected": gate_result.selected_option.value,
                        **gate_result.additional_input,
                    })

                    # Record human gate as executed
                    self.limits.record_execution(agent.name)

                    if gate_result.route == "$end":
                        return self._build_final_output()
                    current_agent_name = gate_result.route
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

                # Record successful execution
                self.limits.record_execution(agent.name)

                # Check timeout after each agent
                self.limits.check_timeout()

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
            Dict with execution statistics including iterations,
            agents executed, context mode, elapsed time, and limits.
        """
        return {
            "iterations": self.limits.current_iteration,
            "agents_executed": self.limits.execution_history.copy(),
            "context_mode": self.config.workflow.context.mode,
            "elapsed_seconds": self.limits.get_elapsed_time(),
            "max_iterations": self.limits.max_iterations,
            "timeout_seconds": self.limits.timeout_seconds,
        }

    def build_execution_plan(self) -> ExecutionPlan:
        """Build an execution plan by analyzing the workflow.

        This traces all possible paths through the workflow without
        actually executing any agents. Used for --dry-run mode.

        Returns:
            ExecutionPlan with steps and possible paths.
        """
        plan = ExecutionPlan(
            workflow_name=self.config.workflow.name,
            entry_point=self.config.workflow.entry_point,
            max_iterations=self.config.workflow.limits.max_iterations,
            timeout_seconds=self.config.workflow.limits.timeout_seconds,
        )

        visited: set[str] = set()
        loop_targets: set[str] = set()

        # Trace from entry_point
        self._trace_path(
            self.config.workflow.entry_point,
            plan,
            visited,
            loop_targets,
        )

        # Mark loop targets in steps
        for step in plan.steps:
            if step.agent_name in loop_targets:
                step.is_loop_target = True

        return plan

    def _trace_path(
        self,
        agent_name: str,
        plan: ExecutionPlan,
        visited: set[str],
        loop_targets: set[str],
    ) -> None:
        """Recursively trace execution path from an agent.

        This method performs a depth-first traversal of the workflow graph,
        building up the execution plan with all reachable agents.

        Args:
            agent_name: Name of the current agent to trace.
            plan: The execution plan being built.
            visited: Set of already visited agent names (to detect loops).
            loop_targets: Set of agents that are targets of loop-back routes.
        """
        if agent_name == "$end":
            return

        agent = self._find_agent(agent_name)
        if agent is None:
            return

        # Check for loop
        is_loop = agent_name in visited
        if is_loop:
            # Mark as loop target and don't recurse further
            loop_targets.add(agent_name)
            return

        visited.add(agent_name)

        # Get routes from the agent (handle both regular agents and human gates)
        routes_info: list[dict[str, Any]] = []
        route_targets: list[str] = []

        if agent.routes:
            for route in agent.routes:
                routes_info.append({
                    "to": route.to,
                    "when": route.when,
                    "is_conditional": route.when is not None,
                })
                route_targets.append(route.to)
        elif agent.options:
            # Human gate with options
            for option in agent.options:
                routes_info.append({
                    "to": option.route,
                    "when": f"selection == '{option.value}'",
                    "is_conditional": True,
                    "label": option.label,
                })
                route_targets.append(option.route)

        # Build step
        step = ExecutionStep(
            agent_name=agent_name,
            agent_type=agent.type or "agent",
            model=agent.model,
            routes=routes_info,
            is_loop_target=False,  # Will be updated after traversal
        )
        plan.steps.append(step)

        # Trace routes
        for target in route_targets:
            if target != "$end":
                self._trace_path(target, plan, visited, loop_targets)
