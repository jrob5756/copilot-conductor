"""Factory for creating agent providers.

This module provides the create_provider factory function for instantiating
the appropriate AgentProvider based on the requested provider type.
"""

from __future__ import annotations

from typing import Any, Literal

from copilot_conductor.exceptions import ProviderError
from copilot_conductor.providers.base import AgentProvider
from copilot_conductor.providers.claude import ANTHROPIC_SDK_AVAILABLE, ClaudeProvider
from copilot_conductor.providers.copilot import CopilotProvider


async def create_provider(
    provider_type: Literal["copilot", "openai-agents", "claude"] = "copilot",
    validate: bool = True,
    mcp_servers: dict[str, Any] | None = None,
) -> AgentProvider:
    """Factory function to create the appropriate provider.

    Creates and optionally validates an AgentProvider instance based on
    the requested provider type. Validation ensures the provider can
    connect to its backend before returning.

    Args:
        provider_type: Which SDK provider to use. Currently only "copilot"
            is fully implemented.
        validate: Whether to validate connection on creation. If True,
            calls validate_connection() and raises ProviderError on failure.
        mcp_servers: MCP server configurations to pass to the provider.

    Returns:
        Configured AgentProvider instance.

    Raises:
        ProviderError: If provider type is unknown or connection validation fails.

    Example:
        >>> provider = await create_provider("copilot")
        >>> # Use provider for agent execution
        >>> await provider.close()
    """
    match provider_type:
        case "copilot":
            provider = CopilotProvider(mcp_servers=mcp_servers)
        case "openai-agents":
            raise ProviderError(
                "OpenAI Agents provider not yet implemented",
                suggestion="Use 'copilot' provider for now",
            )
        case "claude":
            if not ANTHROPIC_SDK_AVAILABLE:
                raise ProviderError(
                    "Claude provider requires anthropic SDK",
                    suggestion="Install with: uv add 'anthropic>=0.77.0,<1.0.0'",
                )
            # ClaudeProvider initialization is synchronous (safe in async context)
            # The SDK client is created in __init__ but actual I/O happens in
            # validate_connection() and execute() which are properly async
            provider = ClaudeProvider()
        case _:
            raise ProviderError(
                f"Unknown provider: {provider_type}",
                suggestion="Valid providers are: copilot, openai-agents, claude",
            )

    if validate and not await provider.validate_connection():
        raise ProviderError(
            f"Failed to connect to {provider_type} provider",
            suggestion="Check your credentials and network connection",
        )

    return provider
