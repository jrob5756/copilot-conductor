"""Typer application definition for Copilot Conductor CLI.

This module defines the main Typer app and global options.
"""

import typer
from rich.console import Console

from copilot_conductor import __version__

# Create the main Typer app
app = typer.Typer(
    name="conductor",
    help="Copilot Conductor - Orchestrate multi-agent workflows defined in YAML.",
    add_completion=False,
)

# Rich console for formatted output
console = Console()


def version_callback(value: bool) -> None:
    """Display version information and exit."""
    if value:
        console.print(f"Copilot Conductor v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Copilot Conductor - Orchestrate multi-agent workflows defined in YAML."""
    pass


# Placeholder command to ensure the app is functional
@app.command()
def run(
    workflow: str = typer.Argument(..., help="Path to the workflow YAML file."),
) -> None:
    """Run a workflow from a YAML file."""
    console.print(f"[bold green]Running workflow:[/bold green] {workflow}")
    console.print("[yellow]Note: Full implementation coming in EPIC-005[/yellow]")
