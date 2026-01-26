"""Implementation of the 'conductor run' command.

This module provides helper functions for executing workflow files.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import typer

from copilot_conductor.config.loader import load_config
from copilot_conductor.engine.workflow import WorkflowEngine
from copilot_conductor.providers.factory import create_provider


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
) -> dict[str, Any]:
    """Execute a workflow asynchronously.

    Args:
        workflow_path: Path to the workflow YAML file.
        inputs: Workflow input values.
        provider_override: Optional provider name to override workflow config.

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
        engine = WorkflowEngine(config, provider)
        result = await engine.run(inputs)
        return result
    finally:
        await provider.close()
