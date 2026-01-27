"""Human gate handler for interactive workflow decisions.

This module implements human-in-the-loop gates that pause workflow execution
for user selection via Rich interactive prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from copilot_conductor.exceptions import HumanGateError
from copilot_conductor.executor.template import TemplateRenderer

if TYPE_CHECKING:
    from copilot_conductor.config.schema import AgentDef, GateOption


@dataclass
class GateResult:
    """Result of a human gate interaction.

    Contains the selected option, the route to take, and any additional
    input collected via prompt_for.
    """

    selected_option: GateOption
    """The option that was selected."""

    route: str
    """The route to take next."""

    additional_input: dict[str, str] = field(default_factory=dict)
    """Any additional text input collected via prompt_for."""


class HumanGateHandler:
    """Handles human-in-the-loop gate interactions.

    This class displays options to the user via Rich-formatted prompts
    and collects their selection. It also supports --skip-gates mode
    for automation testing.

    Example:
        >>> handler = HumanGateHandler()
        >>> result = await handler.handle_gate(agent, context)
        >>> print(f"User selected: {result.selected_option.label}")
        >>> print(f"Routing to: {result.route}")
    """

    def __init__(
        self,
        console: Console | None = None,
        skip_gates: bool = False,
    ) -> None:
        """Initialize the HumanGateHandler.

        Args:
            console: Rich console for output. Creates one if not provided.
            skip_gates: If True, auto-selects first option without prompting.
        """
        self.console = console or Console()
        self.skip_gates = skip_gates
        self.renderer = TemplateRenderer()

    async def handle_gate(
        self,
        agent: AgentDef,
        context: dict[str, Any],
    ) -> GateResult:
        """Handle a human gate interaction.

        Displays the prompt and options to the user, collects their selection,
        and optionally prompts for additional text input.

        Args:
            agent: The human_gate agent definition.
            context: Current workflow context for template rendering.

        Returns:
            GateResult with selected option, route, and any additional input.

        Raises:
            HumanGateError: If gate has no options or interaction fails.
        """
        if not agent.options:
            raise HumanGateError(
                f"Human gate '{agent.name}' has no options defined",
                suggestion="Add 'options' list to the human_gate agent",
            )

        # Render the prompt with context
        prompt_text = self.renderer.render(agent.prompt, context)

        # If skip_gates is enabled, auto-select first option
        if self.skip_gates:
            return self._auto_select(agent.options[0])

        # Display prompt and options, get user selection
        selected = await self._display_and_select(prompt_text, agent.options)

        # Handle prompt_for if specified
        additional_input: dict[str, str] = {}
        if selected.prompt_for:
            additional_input = await self._collect_additional_input(selected.prompt_for)

        return GateResult(
            selected_option=selected,
            route=selected.route,
            additional_input=additional_input,
        )

    async def _display_and_select(
        self,
        prompt_text: str,
        options: list[GateOption],
    ) -> GateOption:
        """Display prompt and get user selection.

        Uses Rich for beautiful terminal UI with numbered options.

        Args:
            prompt_text: The rendered prompt to display.
            options: List of options to choose from.

        Returns:
            The selected GateOption.
        """
        # Display the prompt in a styled panel
        self.console.print()
        self.console.print(
            Panel(
                prompt_text,
                title="[bold cyan]Decision Required[/bold cyan]",
                border_style="cyan",
            )
        )

        # Display options as numbered list
        self.console.print()
        self.console.print("[bold]Options:[/bold]")
        for i, option in enumerate(options, 1):
            self.console.print(f"  [cyan][{i}][/cyan] {option.label}")

        # Get user selection
        valid_choices = [str(i) for i in range(1, len(options) + 1)]
        while True:
            choice = Prompt.ask(
                "\n[bold]Select option[/bold]",
                choices=valid_choices,
                show_choices=True,
            )
            try:
                index = int(choice) - 1
                if 0 <= index < len(options):
                    selected = options[index]
                    self.console.print(
                        f"\n[green]Selected:[/green] {selected.label}"
                    )
                    return selected
            except ValueError:
                pass
            self.console.print("[red]Invalid selection. Please try again.[/red]")

    async def _collect_additional_input(self, field_name: str) -> dict[str, str]:
        """Collect additional text input from user.

        Prompts the user for additional text input as specified by the
        prompt_for field on the selected option.

        Args:
            field_name: The name of the field to prompt for.

        Returns:
            Dictionary with the field name and collected value.
        """
        self.console.print()
        self.console.print(
            f"[bold]Please provide {field_name}:[/bold]"
        )
        value = Prompt.ask(f"  {field_name}")
        return {field_name: value}

    def _auto_select(self, option: GateOption) -> GateResult:
        """Auto-select an option (for --skip-gates mode).

        In automation mode, this method selects the first option without
        user interaction. Useful for CI/CD pipelines and testing.

        Args:
            option: The option to auto-select (usually the first one).

        Returns:
            GateResult with the auto-selected option.
        """
        self.console.print(
            f"\n[dim]Auto-selecting: {option.label} (--skip-gates)[/dim]"
        )
        return GateResult(
            selected_option=option,
            route=option.route,
            additional_input={},  # No input collection in skip mode
        )
