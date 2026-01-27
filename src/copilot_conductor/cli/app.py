"""Typer application definition for Copilot Conductor CLI.

This module defines the main Typer app and global options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from copilot_conductor import __version__

# Create the main Typer app
app = typer.Typer(
    name="conductor",
    help="Copilot Conductor - Orchestrate multi-agent workflows defined in YAML.",
    add_completion=False,
    no_args_is_help=True,
)

# Rich console for formatted output
console = Console(stderr=True)
output_console = Console()


def version_callback(value: bool) -> None:
    """Display version information and exit."""
    if value:
        output_console.print(f"Copilot Conductor v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Copilot Conductor - Orchestrate multi-agent workflows defined in YAML."""
    pass


@app.command()
def run(
    workflow: Annotated[
        Path,
        typer.Argument(
            help="Path to the workflow YAML file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="Override the provider specified in the workflow (e.g., 'copilot').",
        ),
    ] = None,
    raw_inputs: Annotated[
        list[str] | None,
        typer.Option(
            "--input",
            "-i",
            help="Workflow inputs in name=value format. Can be repeated.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show execution plan without running the workflow.",
        ),
    ] = False,
    skip_gates: Annotated[
        bool,
        typer.Option(
            "--skip-gates",
            help="Auto-select first option at human gates (for automation).",
        ),
    ] = False,
) -> None:
    """Run a workflow from a YAML file.

    Execute a multi-agent workflow defined in the specified YAML file.
    Workflow inputs can be provided using --input flags.

    \b
    Examples:
        conductor run workflow.yaml
        conductor run workflow.yaml --input question="What is Python?"
        conductor run workflow.yaml -i question="Hello" -i context="Programming"
        conductor run workflow.yaml --provider copilot
        conductor run workflow.yaml --dry-run
        conductor run workflow.yaml --skip-gates
    """
    import asyncio
    import json

    # Import here to avoid circular imports and defer heavy imports
    from copilot_conductor.cli.run import (
        InputCollector,
        build_dry_run_plan,
        display_execution_plan,
        parse_input_flags,
        run_workflow_async,
    )

    # Handle dry-run mode
    if dry_run:
        try:
            plan = build_dry_run_plan(workflow)
            display_execution_plan(plan, output_console)
            return
        except Exception as e:
            from copilot_conductor.exceptions import ConductorError

            if isinstance(e, ConductorError):
                console.print(f"[bold red]Error:[/bold red] {e}")
            else:
                console.print(f"[bold red]Unexpected error:[/bold red] {e}")
            raise typer.Exit(code=1) from None

    # Collect inputs from both --input and --input.* patterns
    inputs: dict[str, Any] = {}

    # Parse --input name=value style
    if raw_inputs:
        inputs.update(parse_input_flags(raw_inputs))

    # Also parse --input.name=value style from sys.argv
    inputs.update(InputCollector.extract_from_args())

    try:
        # Run the workflow
        result = asyncio.run(run_workflow_async(workflow, inputs, provider, skip_gates))

        # Output as JSON to stdout
        output_console.print_json(json.dumps(result))

    except Exception as e:
        # Import here to avoid circular imports
        from copilot_conductor.exceptions import ConductorError

        if isinstance(e, ConductorError):
            console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        raise typer.Exit(code=1) from None
