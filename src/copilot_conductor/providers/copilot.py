"""GitHub Copilot SDK provider implementation.

This module provides the CopilotProvider class for executing agents
using the GitHub Copilot SDK.

Note: The actual SDK integration will be completed when the github-copilot-sdk
package is available. This implementation provides a functional stub that
can be used for testing and development.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from copilot_conductor.providers.base import AgentOutput, AgentProvider

if TYPE_CHECKING:
    from copilot_conductor.config.schema import AgentDef


class CopilotProvider(AgentProvider):
    """GitHub Copilot SDK provider.

    Translates Conductor agent definitions into Copilot SDK calls and
    normalizes responses into AgentOutput format.

    For testing purposes, this provider supports a mock_handler that can
    be used to simulate agent responses without requiring the actual SDK.

    Example:
        >>> provider = CopilotProvider()
        >>> await provider.validate_connection()
        True
        >>> await provider.close()

        # Using mock handler for testing
        >>> def mock_handler(agent, prompt, context):
        ...     return {"answer": "Mocked response"}
        >>> provider = CopilotProvider(mock_handler=mock_handler)
        >>> output = await provider.execute(agent, {}, "prompt")
        >>> output.content["answer"]
        'Mocked response'
    """

    def __init__(
        self,
        mock_handler: Callable[[AgentDef, str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the Copilot provider.

        Args:
            mock_handler: Optional function that receives (agent, prompt, context)
                         and returns a dict output. Used for testing.
        """
        self._client: Any = None  # Will hold Copilot SDK client
        self._mock_handler = mock_handler
        self._call_history: list[dict[str, Any]] = []

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        tools: list[str] | None = None,
    ) -> AgentOutput:
        """Execute an agent using the Copilot SDK.

        If a mock_handler is configured, it will be used instead of
        the actual SDK. This is useful for testing.

        Args:
            agent: Agent definition from workflow config.
            context: Accumulated workflow context.
            rendered_prompt: Jinja2-rendered user prompt.
            tools: List of tool names available to this agent.

        Returns:
            Normalized AgentOutput with structured content.
        """
        # Record the call for testing purposes
        self._call_history.append({
            "agent_name": agent.name,
            "prompt": rendered_prompt,
            "context": context,
            "tools": tools,
            "model": agent.model,
        })

        # If mock handler is configured, use it
        if self._mock_handler is not None:
            content = self._mock_handler(agent, rendered_prompt, context)
            return AgentOutput(
                content=content,
                raw_response=json.dumps(content),
                tokens_used=0,
                model=agent.model or "mock",
            )

        # Stub implementation for when SDK is not available
        # Returns a response based on the expected output schema
        content = self._generate_stub_output(agent)

        return AgentOutput(
            content=content,
            raw_response=json.dumps(content),
            tokens_used=0,
            model=agent.model or "unknown",
        )

    def _generate_stub_output(self, agent: AgentDef) -> dict[str, Any]:
        """Generate stub output based on agent's output schema.

        Args:
            agent: Agent definition with output schema.

        Returns:
            Dict with stub values matching the schema.
        """
        if not agent.output:
            return {"result": "stub response"}

        result: dict[str, Any] = {}
        for field_name, field_def in agent.output.items():
            result[field_name] = self._generate_stub_value(field_def.type)

        return result

    def _generate_stub_value(self, field_type: str) -> Any:
        """Generate a stub value for a given type.

        Args:
            field_type: The type string (string, number, boolean, array, object).

        Returns:
            A stub value of the appropriate type.
        """
        type_defaults: dict[str, Any] = {
            "string": "stub",
            "number": 0,
            "boolean": True,
            "array": [],
            "object": {},
        }
        return type_defaults.get(field_type, "stub")

    async def validate_connection(self) -> bool:
        """Verify Copilot SDK connection.

        Returns:
            True if connection is valid, False otherwise.

        Note:
            Returns True for stub implementation.
            Real implementation will check SDK authentication.
        """
        # For stub implementation, always return True
        # Real implementation will verify SDK connection
        return True

    async def close(self) -> None:
        """Close Copilot SDK client.

        Releases any resources held by the SDK client.
        """
        self._client = None
        self._call_history.clear()

    def get_call_history(self) -> list[dict[str, Any]]:
        """Get the history of execute calls.

        This is useful for testing to verify that agents were
        called with the expected parameters.

        Returns:
            List of call records with agent_name, prompt, context, tools, model.
        """
        return self._call_history.copy()

    def clear_call_history(self) -> None:
        """Clear the call history."""
        self._call_history.clear()
