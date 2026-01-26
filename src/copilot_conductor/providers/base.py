"""Abstract base class for SDK providers.

This module defines the AgentProvider ABC and AgentOutput dataclass that
all provider implementations must use to ensure a consistent interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from copilot_conductor.config.schema import AgentDef


@dataclass
class AgentOutput:
    """Normalized output from any SDK provider.

    Provides a consistent interface for agent execution results regardless
    of the underlying provider (Copilot, OpenAI, Claude, etc.).

    Attributes:
        content: Parsed structured output matching the agent's output schema.
        raw_response: Provider-specific raw response for debugging/logging.
        tokens_used: Token count if provided by the SDK.
        model: Actual model used (may differ from requested if aliased).
    """

    content: dict[str, Any]
    """Parsed structured output matching the agent's output schema."""

    raw_response: Any
    """Provider-specific raw response for debugging/logging."""

    tokens_used: int | None = None
    """Token count if provided by the SDK."""

    model: str | None = None
    """Actual model used (may differ from requested if aliased)."""


class AgentProvider(ABC):
    """Abstract base class for SDK providers.

    Providers translate between the normalized Conductor interface
    and specific SDK implementations (Copilot, OpenAI, Claude).

    Implementations must provide:
    - execute(): Run an agent and return normalized output
    - validate_connection(): Verify backend connectivity
    - close(): Clean up resources

    Example:
        >>> class MyProvider(AgentProvider):
        ...     async def execute(self, agent, context, rendered_prompt, tools=None):
        ...         # Call SDK and return AgentOutput
        ...         pass
        ...     async def validate_connection(self):
        ...         return True
        ...     async def close(self):
        ...         pass
    """

    @abstractmethod
    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        tools: list[str] | None = None,
    ) -> AgentOutput:
        """Execute an agent and return normalized output.

        Args:
            agent: Agent definition from workflow config.
            context: Accumulated workflow context.
            rendered_prompt: Jinja2-rendered user prompt.
            tools: List of tool names available to this agent.

        Returns:
            Normalized AgentOutput with structured content.

        Raises:
            ProviderError: If SDK execution fails.
            ValidationError: If output doesn't match schema.
        """
        ...

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Verify the provider can connect to its backend.

        This method should perform a lightweight check to ensure the
        provider is properly configured and can reach its backend service.

        Returns:
            True if connection successful, False otherwise.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources and close connections.

        This method should clean up any resources held by the provider,
        such as HTTP clients or session state.
        """
        ...
