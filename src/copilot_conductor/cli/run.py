"""Implementation of the 'conductor run' command.

This module provides helper functions for executing workflow files.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from copilot_conductor.config.loader import load_config
from copilot_conductor.engine.workflow import ExecutionPlan, WorkflowEngine
from copilot_conductor.mcp_auth import resolve_mcp_server_auth
from copilot_conductor.providers.factory import create_provider

if TYPE_CHECKING:
    pass

# Verbose console for logging (stderr)
_verbose_console = Console(stderr=True, highlight=False)


def verbose_log(message: str, style: str = "dim") -> None:
    """Log a message if verbose mode is enabled.

    Args:
        message: The message to log.
        style: Rich style for the message.
    """
    from copilot_conductor.cli.app import is_verbose

    if is_verbose():
        _verbose_console.print(f"[{style}]{message}[/{style}]")


def verbose_log_agent_start(agent_name: str, iteration: int) -> None:
    """Log agent execution start with visual formatting.

    Args:
        agent_name: Name of the agent being executed.
        iteration: Current iteration number (1-indexed).
    """
    from rich.text import Text

    from copilot_conductor.cli.app import is_verbose

    if is_verbose():
        _verbose_console.print()  # Empty line before agent
        text = Text()
        text.append("┌─ ", style="cyan")
        text.append("Agent: ", style="cyan")
        text.append(agent_name, style="cyan bold")
        text.append(f" [iter {iteration}]", style="dim")
        _verbose_console.print(text)


def verbose_log_agent_complete(
    agent_name: str,
    elapsed: float,
    *,
    model: str | None = None,
    tokens: int | None = None,
    output_keys: list[str] | None = None,
) -> None:
    """Log agent completion with summary info.

    Args:
        agent_name: Name of the agent that completed.
        elapsed: Elapsed time in seconds.
        model: Model used (if any).
        tokens: Tokens used (if any).
        output_keys: List of output keys (if dict output).
    """
    from rich.text import Text

    from copilot_conductor.cli.app import is_verbose

    if is_verbose():
        # Build summary line
        parts = [f"{elapsed:.2f}s"]
        if model:
            parts.append(model)
        if tokens:
            parts.append(f"{tokens} tokens")
        if output_keys:
            parts.append(f"→ {output_keys}")

        text = Text()
        text.append("└─ ", style="green")
        text.append("✓ ", style="green")
        text.append(agent_name, style="green")
        text.append(f"  ({', '.join(parts)})", style="dim")
        _verbose_console.print(text)


def verbose_log_route(target: str) -> None:
    """Log routing decision.

    Args:
        target: The routing target.
    """
    from rich.text import Text

    from copilot_conductor.cli.app import is_verbose

    if is_verbose():
        text = Text()
        text.append("   → ", style="yellow")
        if target == "$end":
            text.append("$end", style="yellow bold")
        else:
            text.append("next: ", style="dim")
            text.append(target, style="yellow")
        _verbose_console.print(text)


def verbose_log_section(title: str, content: str, truncate: bool = True) -> None:
    """Log a section with title if verbose mode is enabled.

    Args:
        title: Section title.
        content: Section content.
        truncate: If True, truncate content to 500 chars unless full mode is enabled.
    """
    from copilot_conductor.cli.app import is_full, is_verbose

    if is_verbose():
        display_content = content
        # Truncate content unless full mode is enabled or truncate is False
        if truncate and not is_full() and len(content) > 500:
            display_content = content[:500] + "\n... [truncated, use --verbose for full]"

        _verbose_console.print(
            Panel(display_content, title=f"[cyan]{title}[/cyan]", border_style="dim")
        )


def verbose_log_timing(operation: str, elapsed: float) -> None:
    """Log timing information if verbose mode is enabled.

    Args:
        operation: Description of the operation.
        elapsed: Elapsed time in seconds.
    """
    from copilot_conductor.cli.app import is_verbose

    if is_verbose():
        _verbose_console.print(f"[dim]⏱ {operation}: {elapsed:.2f}s[/dim]")


def verbose_log_parallel_start(group_name: str, agent_count: int) -> None:
    """Log parallel group execution start.

    Args:
        group_name: Name of the parallel group.
        agent_count: Number of agents in the group.
    """
    from rich.text import Text

    from copilot_conductor.cli.app import is_verbose

    if is_verbose():
        text = Text()
        text.append("┌─ ", style="magenta")
        text.append("Parallel Group: ", style="magenta")
        text.append(group_name, style="magenta bold")
        text.append(f" ({agent_count} agents)", style="dim")
        _verbose_console.print()
        _verbose_console.print(text)


def verbose_log_parallel_agent_complete(
    agent_name: str,
    elapsed: float,
    *,
    model: str | None = None,
    tokens: int | None = None,
) -> None:
    """Log parallel agent completion.

    Args:
        agent_name: Name of the agent that completed.
        elapsed: Elapsed time in seconds.
        model: Model used (if any).
        tokens: Tokens used (if any).
    """
    from rich.text import Text

    from copilot_conductor.cli.app import is_verbose

    if is_verbose():
        parts = [f"{elapsed:.2f}s"]
        if model:
            parts.append(model)
        if tokens:
            parts.append(f"{tokens} tokens")

        text = Text()
        text.append("  ✓ ", style="green")
        text.append(agent_name, style="green")
        text.append(f"  ({', '.join(parts)})", style="dim")
        _verbose_console.print(text)


def verbose_log_parallel_agent_failed(
    agent_name: str,
    elapsed: float,
    exception_type: str,
    message: str,
) -> None:
    """Log parallel agent failure.

    Args:
        agent_name: Name of the agent that failed.
        elapsed: Elapsed time in seconds.
        exception_type: Type of exception.
        message: Error message.
    """
    from rich.text import Text

    from copilot_conductor.cli.app import is_verbose

    if is_verbose():
        text = Text()
        text.append("  ✗ ", style="red")
        text.append(agent_name, style="red")
        text.append(f"  ({elapsed:.2f}s)", style="dim")
        _verbose_console.print(text)
        _verbose_console.print(
            f"      {exception_type}: {message}", style="red dim"
        )


def verbose_log_parallel_summary(
    group_name: str,
    success_count: int,
    failure_count: int,
    total_elapsed: float,
) -> None:
    """Log parallel group execution summary.

    Args:
        group_name: Name of the parallel group.
        success_count: Number of agents that succeeded.
        failure_count: Number of agents that failed.
        total_elapsed: Total elapsed time in seconds.
    """
    from rich.text import Text

    from copilot_conductor.cli.app import is_verbose

    if is_verbose():
        text = Text()
        text.append("└─ ", style="cyan")
        
        if failure_count == 0:
            text.append("✓ ", style="green")
            text.append(group_name, style="green")
            text.append(f"  ({success_count}/{success_count} succeeded, {total_elapsed:.2f}s)", style="dim")
        else:
            status_parts = []
            # Always show succeeded count even if 0
            status_parts.append(f"{success_count} succeeded")
            status_parts.append(f"{failure_count} failed")
            
            style = "yellow" if success_count > 0 else "red"
            text.append("◆ ", style=style)
            text.append(group_name, style=style)
            text.append(f"  ({', '.join(status_parts)}, {total_elapsed:.2f}s)", style="dim")
        
        _verbose_console.print(text)


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
    start_time = time.time()

    # Log workflow loading
    verbose_log(f"Loading workflow: {workflow_path}")

    # Load configuration
    load_start = time.time()
    config = load_config(workflow_path)
    verbose_log_timing("Configuration loaded", time.time() - load_start)

    # Log workflow details
    verbose_log(f"Workflow: {config.workflow.name}")
    verbose_log(f"Entry point: {config.workflow.entry_point}")
    verbose_log(f"Agents: {len(config.agents)}")

    if inputs:
        verbose_log_section("Workflow Inputs", json.dumps(inputs, indent=2))

    # Apply provider override if specified
    if provider_override:
        verbose_log(f"Provider override: {provider_override}", style="yellow")
        config.workflow.runtime.provider = provider_override  # type: ignore[assignment]

    # Convert MCP servers from workflow config to SDK format
    mcp_servers: dict[str, Any] | None = None
    if config.workflow.runtime.mcp_servers:
        mcp_servers = {}
        for name, server in config.workflow.runtime.mcp_servers.items():
            # Convert Pydantic model to dict for SDK
            if server.type in ("http", "sse"):
                server_config: dict[str, Any] = {
                    "type": server.type,
                    "url": server.url,
                    "tools": server.tools,
                }
                if server.headers:
                    server_config["headers"] = server.headers
                if server.timeout:
                    server_config["timeout"] = server.timeout
                # Resolve OAuth authentication for HTTP/SSE servers
                server_config = await resolve_mcp_server_auth(name, server_config)
            else:
                # stdio/local type
                server_config = {
                    "type": "stdio",
                    "command": server.command,
                    "args": server.args,
                    "tools": server.tools,
                }
                if server.env:
                    server_config["env"] = server.env
                if server.timeout:
                    server_config["timeout"] = server.timeout
            mcp_servers[name] = server_config
        verbose_log(f"MCP servers configured: {list(mcp_servers.keys())}")

    # Create provider
    verbose_log(f"Creating provider: {config.workflow.runtime.provider}")
    provider = await create_provider(
        config.workflow.runtime.provider,
        mcp_servers=mcp_servers,
    )

    try:
        # Create and run workflow engine
        verbose_log("Starting workflow execution...")

        engine = WorkflowEngine(config, provider, skip_gates=skip_gates)
        result = await engine.run(inputs)

        # Log completion
        verbose_log_timing("Total workflow execution", time.time() - start_time)
        verbose_log("Workflow completed successfully", style="green")

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
    timeout_display = f"{plan.timeout_seconds}s" if plan.timeout_seconds else "unlimited"
    header_content = (
        f"[bold]Workflow:[/bold] {plan.workflow_name}\n"
        f"[bold]Entry Point:[/bold] {plan.entry_point}\n"
        f"[bold]Max Iterations:[/bold] {plan.max_iterations}\n"
        f"[bold]Timeout:[/bold] {timeout_display}"
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
        
        # Handle parallel groups differently
        if step.agent_type == "parallel_group":
            # Show parallel group with failure mode
            failure_mode_display = step.failure_mode or "fail_fast"
            model_info = f"[dim]{failure_mode_display}[/dim]"
            
            table.add_row(
                str(i),
                f"{step.agent_name}{loop_marker}",
                step.agent_type,
                model_info,
                routes_str,
            )
            
            # Add a detail row showing which agents execute in parallel
            if step.parallel_agents:
                agents_display = ", ".join(f"[cyan]{agent}[/cyan]" for agent in step.parallel_agents)
                table.add_row(
                    "",
                    f"[dim]  ⚡ {agents_display}[/dim]",
                    "",
                    "",
                    "",
                )
        else:
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
    parallel_group_count = sum(1 for s in plan.steps if s.agent_type == "parallel_group")
    total_parallel_agents = sum(
        len(s.parallel_agents or []) 
        for s in plan.steps 
        if s.agent_type == "parallel_group"
    )
    
    summary_parts = [
        f"[dim]Total steps:[/dim] {len(plan.steps)}",
        f"[dim]Loop targets:[/dim] {sum(1 for s in plan.steps if s.is_loop_target)}",
    ]
    
    if parallel_group_count > 0:
        summary_parts.append(f"[dim]Parallel groups:[/dim] {parallel_group_count}")
        summary_parts.append(f"[dim]Parallel agents:[/dim] {total_parallel_agents}")
    
    output_console.print(" | ".join(summary_parts))


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
