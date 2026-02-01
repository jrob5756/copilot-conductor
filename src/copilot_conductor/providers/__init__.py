"""Providers module for Copilot Conductor.

This module defines the agent provider abstraction and implementations
for different LLM providers (Copilot SDK, Claude SDK, etc.).
"""

from copilot_conductor.providers.base import AgentOutput, AgentProvider
from copilot_conductor.providers.claude import ClaudeProvider
from copilot_conductor.providers.copilot import CopilotProvider
from copilot_conductor.providers.factory import create_provider

__all__ = [
    "AgentOutput",
    "AgentProvider",
    "ClaudeProvider",
    "CopilotProvider",
    "create_provider",
]
