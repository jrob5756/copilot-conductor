"""Implementation of the 'conductor run' command.

This module provides helper functions for executing workflow files.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from copilot_conductor.config.loader import load_config
from copilot_conductor.engine.workflow import ExecutionPlan, WorkflowEngine
from copilot_conductor.providers.factory import create_provider

if TYPE_CHECKING:
    pass


def parse_input_flags(raw_inputs: list[str]) -> dict[str, Any]:
    """Parse --input.<name>=<value> flags into a dictionary.

    Supports type coercion for common types:
    - "true"/"false" -> bool
    - numeric strings -> int/float
    - JSON arrays/objects -> parsed JSON
    - everything else -> string

    Args:
        raw_inputs: List of "name=value" strings from CLI.

    Returns:
        Dictionary of parsed input name-value pairs.

    Raises:
        typer.BadParameter: If input format is invalid.
    """
    inputs: dict[str, Any] = {}

    for raw in raw_inputs:
        # Split on first = only
        if "=" not in raw:
            raise typer.BadParameter(
                f"Invalid input format: '{raw}'. Expected format: name=value"
            )

        name, value = raw.split("=", 1)
        name = name.strip()
        value = value.strip()

        if not name:
            raise typer.BadParameter(f"Empty input name in: '{raw}'")

        # Type coercion
        inputs[name] = coerce_value(value)

    return inputs


def coerce_value(value: str) -> Any:
    """Coerce a string value to an appropriate Python type.

    Args:
        value: The string value to coerce.

    Returns:
        The coerced value (bool, int, float, list, dict, or str).
    """
    # Handle booleans
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Handle null
    if value.lower() == "null":
        return None

    # Try JSON for arrays and objects
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # Try numeric conversion
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # Return as string
    return value


class InputCollector:
    """Collects input values from --input.* options.

    This class handles parsing of dynamic input options that follow
    the pattern --input.<name>=<value>.
    """

    INPUT_PATTERN = re.compile(r"^--input\.(.+)$")

    @classmethod
    def extract_from_args(cls, args: list[str] | None = None) -> dict[str, Any]:
        """Extract input values from command line arguments.

        Scans sys.argv (or provided args) for --input.* patterns and
        extracts their values.

        Args:
            args: Optional list of arguments to parse. Defaults to sys.argv.

        Returns:
            Dictionary of input name-value pairs.
        """
        if args is None:
            args = sys.argv[1:]

        inputs: dict[str, Any] = {}
        i = 0
        while i < len(args):
            arg = args[i]
            match = cls.INPUT_PATTERN.match(arg)

            if match:
                name = match.group(1)

                # Check for = in the argument (--input.name=value)
                if "=" in name:
                    name, value = name.split("=", 1)
                    inputs[name] = coerce_value(value)
                elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                    # Next argument is the value
                    value = args[i + 1]
                    inputs[name] = coerce_value(value)
                    i += 1
                else:
                    # Boolean flag style (presence = true)
                    inputs[name] = True

            i += 1

        return inputs


async def run_workflow_async(
    workflow_path: Path,
    inputs: dict[str, Any],
    provider_override: str | None = None,
    skip_gates: bool = False,
) -> dict[str, Any]:
    """Execute a workflow asynchronously.

    Args:
        workflow_path: Path to the workflow YAML file.
        inputs: Workflow input values.
        provider_override: Optional provider name to override workflow config.
        skip_gates: If True, auto-selects first option at human gates.

    Returns:
        The workflow output as a dictionary.

    Raises:
        ConductorError: If workflow execution fails.
    """
    # Load configuration
    config = load_config(workflow_path)

    # Apply provider override if specified
    if provider_override:
        config.workflow.runtime.provider = provider_override  # type: ignore[assignment]

    # Create provider
    provider = await create_provider(config.workflow.runtime.provider)

    try:
        # Create and run workflow engine
        engine = WorkflowEngine(config, provider, skip_gates=skip_gates)
        result = await engine.run(inputs)
        return result
    finally:
        await provider.close()


def format_routes(routes: list[dict[str, Any]]) -> str:
    """Format routes for display in the dry-run table.

    Args:
        routes: List of route dictionaries with 'to', 'when', and 'is_conditional' keys.

    Returns:
        Formatted string representation of routes.
    """
    if not routes:
        return "[dim]$end[/dim]"

    parts = []
    for route in routes:
        if route.get("is_conditional"):
            condition = route.get("when", "?")
            # Truncate long conditions
            if len(condition) > 40:
                condition = condition[:37] + "..."
            parts.append(f"→ {route['to']} [dim](if {condition})[/dim]")
        else:
            parts.append(f"→ {route['to']}")
    return "\n".join(parts) if parts else "[dim]$end[/dim]"


def display_execution_plan(plan: ExecutionPlan, console: Console | None = None) -> None:
    """Display execution plan with Rich formatting.

    Renders a formatted view of the execution plan including workflow
    metadata, agent sequence with models, and routing information.

    Args:
        plan: The execution plan to display.
        console: Optional Rich console. Creates one if not provided.
    """
    output_console = console if console is not None else Console()

    # Header panel with workflow metadata
    header_content = (
        f"[bold]Workflow:[/bold] {plan.workflow_name}\n"
        f"[bold]Entry Point:[/bold] {plan.entry_point}\n"
        f"[bold]Max Iterations:[/bold] {plan.max_iterations}\n"
        f"[bold]Timeout:[/bold] {plan.timeout_seconds}s"
    )
    output_console.print(Panel(header_content, title="[cyan]Execution Plan (Dry Run)[/cyan]"))

    # Steps table
    table = Table(title="Agent Sequence", show_lines=True)
    table.add_column("Step", style="cyan", justify="right", width=6)
    table.add_column("Agent", style="green")
    table.add_column("Type", width=12)
    table.add_column("Model", width=20)
    table.add_column("Routes")

    for i, step in enumerate(plan.steps, 1):
        routes_str = format_routes(step.routes)
        loop_marker = " [yellow](loop target)[/yellow]" if step.is_loop_target else ""

        table.add_row(
            str(i),
            f"{step.agent_name}{loop_marker}",
            step.agent_type,
            step.model or "[dim]default[/dim]",
            routes_str,
        )

    output_console.print(table)

    # Print summary
    output_console.print()
    output_console.print(
        f"[dim]Total agents:[/dim] {len(plan.steps)} | "
        f"[dim]Loop targets:[/dim] {sum(1 for s in plan.steps if s.is_loop_target)}"
    )


def build_dry_run_plan(workflow_path: Path) -> ExecutionPlan:
    """Build an execution plan for dry-run mode.

    Loads the workflow configuration and builds an execution plan
    without creating a provider or executing any agents.

    Args:
        workflow_path: Path to the workflow YAML file.

    Returns:
        ExecutionPlan showing the workflow structure.
    """
    # Load configuration
    config = load_config(workflow_path)

    # Create engine without provider (we won't execute anything)
    # We need a dummy provider for the constructor, but we won't use it
    # Instead, we'll create a minimal WorkflowEngine-like object
    # Actually, let's refactor to allow None provider for dry-run

    # For now, we'll create a minimal engine setup
    from copilot_conductor.engine.context import WorkflowContext
    from copilot_conductor.engine.limits import LimitEnforcer
    from copilot_conductor.engine.router import Router
    from copilot_conductor.executor.template import TemplateRenderer

    # Create a partial engine with just what we need for plan building
    class _DryRunEngine:
        def __init__(self, cfg: Any) -> None:
            self.config = cfg
            self.context = WorkflowContext()
            self.renderer = TemplateRenderer()
            self.router = Router()
            self.limits = LimitEnforcer(
                max_iterations=cfg.workflow.limits.max_iterations,
                timeout_seconds=cfg.workflow.limits.timeout_seconds,
            )

        def _find_agent(self, name: str) -> Any:
            return next((a for a in self.config.agents if a.name == name), None)

    # Use a real WorkflowEngine but with a mock provider
    from copilot_conductor.config.schema import AgentDef
    from copilot_conductor.providers.base import AgentOutput, AgentProvider

    class _MockProvider(AgentProvider):
        async def execute(
            self,
            agent: AgentDef,
            context: dict[str, Any],
            rendered_prompt: str,
            tools: list[str] | None = None,
        ) -> AgentOutput:
            return AgentOutput(content={}, raw_response="")

        async def validate_connection(self) -> bool:
            return True

        async def close(self) -> None:
            pass

    engine = WorkflowEngine(config, _MockProvider())
    return engine.build_execution_plan()
