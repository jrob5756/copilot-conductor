"""Cross-field validators for workflow configuration.

This module provides additional validation beyond Pydantic schema validation,
including semantic checks for agent references, input dependencies, and
tool references.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from copilot_conductor.exceptions import ConfigurationError

if TYPE_CHECKING:
    from copilot_conductor.config.schema import WorkflowConfig


# Pattern for input references: agent.output, workflow.input.param, with optional ? suffix
INPUT_REF_PATTERN = re.compile(
    r"^(?:(?P<agent>[a-zA-Z_][a-zA-Z0-9_]*)\.output(?:\.(?P<field>[a-zA-Z_][a-zA-Z0-9_]*))?|"
    r"workflow\.input\.(?P<input>[a-zA-Z_][a-zA-Z0-9_]*))(?P<optional>\?)?$"
)


def validate_workflow_config(config: WorkflowConfig) -> list[str]:
    """Perform comprehensive validation of a workflow configuration.

    This function performs semantic validation beyond what Pydantic can check,
    including cross-field references and consistency checks.

    Args:
        config: The WorkflowConfig to validate.

    Returns:
        A list of warning messages (non-fatal issues).

    Raises:
        ConfigurationError: If any validation errors are found.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Build index of agent names
    agent_names = {agent.name for agent in config.agents}

    # Validate entry_point exists (already done by Pydantic, but good to have explicit)
    if config.workflow.entry_point not in agent_names:
        errors.append(
            f"entry_point '{config.workflow.entry_point}' not found in agents. "
            f"Available agents: {', '.join(sorted(agent_names))}"
        )

    # Validate each agent
    for agent in config.agents:
        # Validate route targets
        agent_errors = _validate_agent_routes(agent.name, agent.routes, agent_names)
        errors.extend(agent_errors)

        # Validate human_gate has options
        if agent.type == "human_gate":
            if not agent.options:
                errors.append(f"Agent '{agent.name}' is a human_gate but has no options defined")
            else:
                # Validate gate option routes
                for i, option in enumerate(agent.options):
                    if option.route != "$end" and option.route not in agent_names:
                        errors.append(
                            f"Agent '{agent.name}' gate option {i} ('{option.label}') "
                            f"routes to unknown agent '{option.route}'"
                        )

        # Validate input references
        input_errors, input_warnings = _validate_input_references(
            agent.name,
            agent.input,
            agent_names,
            set(config.workflow.input.keys()),
        )
        errors.extend(input_errors)
        warnings.extend(input_warnings)

        # Validate tool references
        if agent.tools is not None and agent.tools:
            tool_errors = _validate_tool_references(
                agent.name, agent.tools, set(config.tools)
            )
            errors.extend(tool_errors)

    # Validate workflow output references
    output_errors = _validate_output_references(
        config.output, agent_names, set(config.workflow.input.keys())
    )
    errors.extend(output_errors)

    if errors:
        raise ConfigurationError(
            "Workflow configuration validation failed:\n  - " + "\n  - ".join(errors),
            suggestion="Fix the validation errors listed above and try again.",
        )

    return warnings


def _validate_agent_routes(
    agent_name: str,
    routes: list,
    agent_names: set[str],
) -> list[str]:
    """Validate that all route targets exist.

    Args:
        agent_name: Name of the agent whose routes are being validated.
        routes: List of RouteDef objects.
        agent_names: Set of valid agent names.

    Returns:
        List of error messages.
    """
    errors: list[str] = []

    for i, route in enumerate(routes):
        if route.to != "$end" and route.to not in agent_names:
            errors.append(
                f"Agent '{agent_name}' route {i} targets unknown agent '{route.to}'. "
                f"Use '$end' to terminate or one of: {', '.join(sorted(agent_names))}"
            )

    return errors


def _validate_input_references(
    agent_name: str,
    inputs: list[str],
    agent_names: set[str],
    workflow_inputs: set[str],
) -> tuple[list[str], list[str]]:
    """Validate input reference formats and targets.

    Args:
        agent_name: Name of the agent whose inputs are being validated.
        inputs: List of input reference strings.
        agent_names: Set of valid agent names.
        workflow_inputs: Set of valid workflow input parameter names.

    Returns:
        Tuple of (error messages, warning messages).
    """
    errors: list[str] = []
    warnings: list[str] = []

    for input_ref in inputs:
        match = INPUT_REF_PATTERN.match(input_ref)

        if not match:
            errors.append(
                f"Agent '{agent_name}' has invalid input reference '{input_ref}'. "
                "Expected format: 'agent_name.output', 'agent_name.output.field', "
                "or 'workflow.input.param_name' (append '?' for optional)"
            )
            continue

        # Check if referencing another agent's output
        ref_agent = match.group("agent")
        if ref_agent and ref_agent not in agent_names:
            is_optional = match.group("optional") == "?"
            if is_optional:
                warnings.append(
                    f"Agent '{agent_name}' has optional reference to "
                    f"unknown agent '{ref_agent}'"
                )
            else:
                errors.append(
                    f"Agent '{agent_name}' references unknown agent '{ref_agent}' in input"
                )

        # Check if referencing workflow input
        workflow_input = match.group("input")
        if workflow_input and workflow_input not in workflow_inputs:
            is_optional = match.group("optional") == "?"
            if is_optional:
                warnings.append(
                    f"Agent '{agent_name}' has optional reference to unknown "
                    f"workflow input '{workflow_input}'"
                )
            else:
                errors.append(
                    f"Agent '{agent_name}' references unknown workflow input "
                    f"'{workflow_input}'. Available: {', '.join(sorted(workflow_inputs))}"
                )

    return errors, warnings


def _validate_tool_references(
    agent_name: str,
    agent_tools: list[str],
    workflow_tools: set[str],
) -> list[str]:
    """Validate that agent tools are defined at workflow level.

    Args:
        agent_name: Name of the agent whose tools are being validated.
        agent_tools: List of tool names the agent wants to use.
        workflow_tools: Set of tools defined at workflow level.

    Returns:
        List of error messages.
    """
    errors: list[str] = []

    for tool in agent_tools:
        if tool not in workflow_tools:
            errors.append(
                f"Agent '{agent_name}' references unknown tool '{tool}'. "
                f"Available tools: {', '.join(sorted(workflow_tools))}"
            )

    return errors


def _validate_output_references(
    output: dict[str, str],
    agent_names: set[str],
    workflow_inputs: set[str],
) -> list[str]:
    """Validate that output template references are valid.

    This performs a basic check for obvious references in Jinja2 templates.
    Full validation happens at render time.

    Args:
        output: Dict of output field names to template expressions.
        agent_names: Set of valid agent names.
        workflow_inputs: Set of valid workflow input parameter names.

    Returns:
        List of error messages.
    """
    # This is a basic check - full validation happens at render time
    # We just check for obvious issues in the template patterns
    errors: list[str] = []

    # Pattern to find potential agent references in templates
    agent_ref_pattern = re.compile(r"\{\{\s*(\w+)\.output")

    for field, template in output.items():
        matches = agent_ref_pattern.findall(template)
        for ref in matches:
            if ref not in agent_names and ref not in ("workflow", "context"):
                errors.append(
                    f"Workflow output '{field}' references unknown agent '{ref}'"
                )

    return errors
