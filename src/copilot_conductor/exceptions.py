"""Exception hierarchy for Copilot Conductor.

This module defines all custom exceptions used throughout the application.
All exceptions inherit from ConductorError and support optional suggestions
to help users resolve issues.
"""

from __future__ import annotations


class ConductorError(Exception):
    """Base exception for all Copilot Conductor errors.

    All custom exceptions in the application inherit from this class.
    Supports an optional suggestion message to help users resolve the issue.

    Attributes:
        suggestion: Optional actionable advice for resolving the error.
    """

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        """Initialize a ConductorError.

        Args:
            message: The error message describing what went wrong.
            suggestion: Optional advice for resolving the error.
        """
        self.suggestion = suggestion
        super().__init__(message)

    def __str__(self) -> str:
        """Format the error message with optional suggestion."""
        msg = super().__str__()
        if self.suggestion:
            msg += f"\n\n💡 Suggestion: {self.suggestion}"
        return msg


class ConfigurationError(ConductorError):
    """Raised when workflow configuration is invalid.

    This includes malformed YAML, missing required fields, or invalid
    configuration values.
    """

    pass


class ValidationError(ConductorError):
    """Raised when data validation fails.

    This includes Pydantic validation errors, schema mismatches, and
    cross-field validation failures.
    """

    pass


class TemplateError(ConductorError):
    """Raised when Jinja2 template rendering fails.

    This includes undefined variables, syntax errors, and filter errors
    in template expressions.
    """

    pass


class ProviderError(ConductorError):
    """Raised when an agent provider encounters an error.

    This includes SDK initialization failures, API errors, and
    connection issues with the underlying provider.
    """

    pass


class ExecutionError(ConductorError):
    """Raised when workflow execution fails.

    Base class for execution-related errors. More specific execution
    errors inherit from this class.
    """

    pass


class MaxIterationsError(ExecutionError):
    """Raised when a workflow exceeds its maximum iteration limit.

    This is a safety mechanism to prevent infinite loops in workflows
    with loop-back routing patterns.

    Attributes:
        iterations: The number of iterations that were executed.
        max_iterations: The configured maximum number of iterations.
        agent_history: List of agents that were executed before the limit.
    """

    def __init__(
        self,
        message: str,
        *,
        iterations: int,
        max_iterations: int,
        agent_history: list[str] | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize a MaxIterationsError.

        Args:
            message: The error message describing what went wrong.
            iterations: The number of iterations that were executed.
            max_iterations: The configured maximum number of iterations.
            agent_history: List of agent names executed before the limit.
            suggestion: Optional advice for resolving the error.
        """
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.agent_history = agent_history or []
        super().__init__(message, suggestion)


class TimeoutError(ExecutionError):
    """Raised when a workflow exceeds its timeout limit.

    This is a safety mechanism to prevent workflows from running
    indefinitely.

    Attributes:
        elapsed_seconds: The time elapsed before the timeout.
        timeout_seconds: The configured timeout limit.
        current_agent: The agent that was executing when timeout occurred.
    """

    def __init__(
        self,
        message: str,
        *,
        elapsed_seconds: float,
        timeout_seconds: float,
        current_agent: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize a TimeoutError.

        Args:
            message: The error message describing what went wrong.
            elapsed_seconds: The time elapsed before the timeout.
            timeout_seconds: The configured timeout limit.
            current_agent: The agent that was executing when timeout occurred.
            suggestion: Optional advice for resolving the error.
        """
        self.elapsed_seconds = elapsed_seconds
        self.timeout_seconds = timeout_seconds
        self.current_agent = current_agent
        super().__init__(message, suggestion)


class HumanGateError(ExecutionError):
    """Raised when a human gate encounters an error.

    This includes invalid gate configurations, user cancellation,
    and input validation failures at human gates.
    """

    pass
