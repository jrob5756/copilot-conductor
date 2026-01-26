"""GitHub Copilot SDK provider implementation.

This module provides the CopilotProvider class for executing agents
using the GitHub Copilot SDK.

Note: This is a stub implementation. Full SDK integration will be
completed in EPIC-004 (TASK-022).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from copilot_conductor.providers.base import AgentOutput, AgentProvider

if TYPE_CHECKING:
    from copilot_conductor.config.schema import AgentDef


class CopilotProvider(AgentProvider):
    """GitHub Copilot SDK provider.

    Translates Conductor agent definitions into Copilot SDK calls and
    normalizes responses into AgentOutput format.

    Note: This is a stub. Full implementation with SDK integration
    will be completed in EPIC-004 (TASK-022).

    Example:
        >>> provider = CopilotProvider()
        >>> await provider.validate_connection()
        True
        >>> await provider.close()
    """

    def __init__(self) -> None:
        """Initialize the Copilot provider.

        The actual SDK client will be initialized here in the full
        implementation.
        """
        self._client: Any = None  # Will hold Copilot SDK client

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        tools: list[str] | None = None,
    ) -> AgentOutput:
        """Execute an agent using the Copilot SDK.

        Args:
            agent: Agent definition from workflow config.
            context: Accumulated workflow context.
            rendered_prompt: Jinja2-rendered user prompt.
            tools: List of tool names available to this agent.

        Returns:
            Normalized AgentOutput with structured content.

        Note:
            This is a stub implementation that returns mock output.
            Real implementation in TASK-022 will call the Copilot SDK.
        """
        # Stub implementation - returns mock output
        # Real implementation in TASK-022
        return AgentOutput(
            content={"result": "stub response"},
            raw_response=None,
            tokens_used=0,
            model=agent.model or "unknown",
        )

    async def validate_connection(self) -> bool:
        """Verify Copilot SDK connection.

        Returns:
            True if connection is valid, False otherwise.

        Note:
            Stub implementation always returns True.
            Real implementation will check SDK authentication.
        """
        # Stub - always returns True for now
        # Real implementation will check SDK auth
        return True

    async def close(self) -> None:
        """Close Copilot SDK client.

        Releases any resources held by the SDK client.
        """
        self._client = None
