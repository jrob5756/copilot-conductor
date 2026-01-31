"""GitHub Copilot SDK provider implementation.

This module provides the CopilotProvider class for executing agents
using the GitHub Copilot SDK.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from copilot_conductor.exceptions import ProviderError
from copilot_conductor.providers.base import AgentOutput, AgentProvider

if TYPE_CHECKING:
    from copilot_conductor.config.schema import AgentDef

# Try to import the Copilot SDK
try:
    from copilot import CopilotClient

    COPILOT_SDK_AVAILABLE = True
except ImportError:
    COPILOT_SDK_AVAILABLE = False
    CopilotClient = None  # type: ignore[misc, assignment]


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (including first attempt).
        base_delay: Base delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        jitter: Maximum random jitter to add to delay (0.0 to 1.0 fraction of delay).
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.25


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
        retry_config: RetryConfig | None = None,
        model: str | None = None,
        mcp_servers: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Copilot provider.

        Args:
            mock_handler: Optional function that receives (agent, prompt, context)
                         and returns a dict output. Used for testing.
            retry_config: Optional retry configuration. Uses default if not provided.
            model: Default model to use if not specified in agent. Defaults to "gpt-4o".
            mcp_servers: MCP server configurations to pass to the SDK.
        """
        self._client: Any = None  # Will hold Copilot SDK client
        self._mock_handler = mock_handler
        self._call_history: list[dict[str, Any]] = []
        self._retry_config = retry_config or RetryConfig()
        self._retry_history: list[dict[str, Any]] = []  # For testing retries
        self._default_model = model or "gpt-4o"
        self._mcp_servers = mcp_servers or {}
        self._started = False

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

        Raises:
            ProviderError: If execution fails after all retry attempts.
        """
        # Record the call for testing purposes
        self._call_history.append({
            "agent_name": agent.name,
            "prompt": rendered_prompt,
            "context": context,
            "tools": tools,
            "model": agent.model,
        })

        # Use retry logic for both mock and real SDK calls
        return await self._execute_with_retry(
            agent, context, rendered_prompt, tools
        )

    async def _execute_with_retry(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        tools: list[str] | None = None,
    ) -> AgentOutput:
        """Execute with exponential backoff retry logic.

        Args:
            agent: Agent definition from workflow config.
            context: Accumulated workflow context.
            rendered_prompt: Jinja2-rendered user prompt.
            tools: List of tool names available to this agent.

        Returns:
            Normalized AgentOutput with structured content.

        Raises:
            ProviderError: If execution fails after all retry attempts.
        """
        last_error: Exception | None = None
        config = self._retry_config

        for attempt in range(1, config.max_attempts + 1):
            try:
                content = await self._execute_sdk_call(
                    agent, rendered_prompt, context, tools
                )
                return AgentOutput(
                    content=content,
                    raw_response=json.dumps(content),
                    tokens_used=0,
                    model=agent.model or "mock",
                )
            except ProviderError as e:
                last_error = e
                self._retry_history.append({
                    "attempt": attempt,
                    "agent_name": agent.name,
                    "error": str(e),
                    "is_retryable": e.is_retryable,
                })

                # Don't retry non-retryable errors
                if not e.is_retryable:
                    raise

                # Don't retry if this was the last attempt
                if attempt >= config.max_attempts:
                    break

                # Calculate delay with exponential backoff
                delay = self._calculate_delay(attempt, config)

                # Log retry attempt (for testing visibility)
                self._retry_history[-1]["delay"] = delay

                await asyncio.sleep(delay)

            except Exception as e:
                # Wrap unexpected errors as retryable
                last_error = e
                self._retry_history.append({
                    "attempt": attempt,
                    "agent_name": agent.name,
                    "error": str(e),
                    "is_retryable": True,
                })

                if attempt >= config.max_attempts:
                    break

                delay = self._calculate_delay(attempt, config)
                self._retry_history[-1]["delay"] = delay
                await asyncio.sleep(delay)

        # All retries exhausted
        raise ProviderError(
            f"SDK call failed after {config.max_attempts} attempts: {last_error}",
            suggestion=f"Check provider configuration and connectivity. Last error: {last_error}",
            is_retryable=False,
        )

    async def _execute_sdk_call(
        self,
        agent: AgentDef,
        rendered_prompt: str,
        context: dict[str, Any],
        tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute the actual SDK call or mock handler.

        Args:
            agent: Agent definition from workflow config.
            rendered_prompt: Jinja2-rendered user prompt.
            context: Accumulated workflow context.
            tools: List of tool names available to this agent.

        Returns:
            Dict containing the agent's response content.

        Raises:
            ProviderError: If the SDK call fails.
        """
        if self._mock_handler is not None:
            # Mock handler for testing
            return self._mock_handler(agent, rendered_prompt, context)

        # Use the real Copilot SDK
        if not COPILOT_SDK_AVAILABLE:
            raise ProviderError(
                "GitHub Copilot SDK is not installed",
                suggestion="Install with: pip install github-copilot-sdk",
                is_retryable=False,
            )

        # Ensure client is started
        if not self._started:
            await self._ensure_client_started()

        model = agent.model or self._default_model

        # Build the full prompt with system prompt if provided
        full_prompt = rendered_prompt
        if agent.system_prompt:
            full_prompt = f"System: {agent.system_prompt}\n\nUser: {rendered_prompt}"

        # If agent has output schema, instruct to respond in JSON
        if agent.output:
            schema_desc = json.dumps(
                {
                    name: {
                        "type": field.type,
                        "description": field.description or f"The {name} field",
                    }
                    for name, field in agent.output.items()
                },
                indent=2,
            )
            full_prompt += (
                f"\n\n**IMPORTANT: You MUST respond with a JSON object matching this schema:**\n"
                f"```json\n{schema_desc}\n```\n"
                f"Return ONLY the JSON object, no other text."
            )

        try:
            # Build session config with MCP servers from workflow configuration
            session_config: dict[str, Any] = {
                "model": model,
            }

            # Add MCP servers if configured
            if self._mcp_servers:
                session_config["mcp_servers"] = self._mcp_servers

            # Create a session and send the prompt
            session = await self._client.create_session(session_config)

            # Collect response using event handler
            response_content = ""
            done = asyncio.Event()
            error_message: str | None = None

            # Capture verbose state before callback (contextvars don't propagate to sync callbacks)
            from copilot_conductor.cli.app import is_full, is_verbose
            verbose_enabled = is_verbose()
            full_enabled = is_full()

            def on_event(event: Any) -> None:
                nonlocal response_content, error_message
                event_type = event.type.value if hasattr(event.type, "value") else str(event.type)

                if event_type == "assistant.message":
                    response_content = event.data.content
                elif event_type == "session.idle":
                    done.set()
                elif event_type == "error" or event_type == "session.error":
                    error_message = getattr(event.data, "message", str(event.data))
                    done.set()

                # Verbose logging for intermediate progress
                if verbose_enabled:
                    self._log_event_verbose(event_type, event, full_enabled)

            session.on(on_event)
            await session.send({"prompt": full_prompt})

            # Wait for completion - no timeout at SDK level
            # The workflow-level timeout will enforce overall limits
            await done.wait()

            await session.destroy()

            if error_message:
                raise ProviderError(
                    f"Copilot SDK error: {error_message}",
                    is_retryable=True,
                )

            # Parse the response as JSON if we have an output schema
            if agent.output and response_content:
                try:
                    # Try to extract JSON from the response
                    return self._extract_json(response_content)
                except (json.JSONDecodeError, ValueError) as e:
                    # Provide better error for structured output parsing failures
                    expected_fields = list(agent.output.keys())
                    raise ProviderError(
                        f"Failed to parse structured output from agent response: {e}",
                        suggestion=(
                            f"Agent was expected to return JSON with fields: {expected_fields}. "
                            f"Response started with: {response_content[:200]}..."
                        ),
                        is_retryable=True,
                    ) from e

            return {"result": response_content}

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                f"Copilot SDK call failed: {e}",
                suggestion="Check that copilot CLI is installed and authenticated",
                is_retryable=True,
            ) from e

    def _extract_json(self, content: str) -> dict[str, Any]:
        """Extract JSON from response content.

        Handles responses that may have markdown code blocks or extra text.

        Args:
            content: The response content string.

        Returns:
            Parsed JSON as dict.

        Raises:
            ValueError: If no valid JSON found.
        """
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code blocks
        import re

        # Look for ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Look for {...} pattern
        brace_match = re.search(r"\{.*\}", content, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON from response: {content[:200]}...")

    def _log_event_verbose(self, event_type: str, event: Any, full_mode: bool) -> None:
        """Log SDK events in verbose mode for progress visibility.

        Note: Caller must check is_verbose() before calling - contextvars
        don't propagate to sync callbacks from the SDK.

        Args:
            event_type: The event type string.
            event: The event object.
            full_mode: If True, show full details (args, results, reasoning).
        """
        from rich.console import Console
        from rich.text import Text

        console = Console(stderr=True, highlight=False)

        # Log interesting events with Rich styling
        if event_type == "tool.execution_start":
            tool_name = (
                getattr(event.data, "tool_name", None)
                or getattr(event.data, "name", "unknown")
            )

            text = Text()
            text.append("    ├─ ", style="dim")
            text.append("🔧 ", style="")
            text.append(tool_name, style="cyan bold")
            console.print(text)

            # In full mode, try to show arguments
            if full_mode:
                args = getattr(event.data, "arguments", None) or getattr(event.data, "args", None)
                if args:
                    args_str = str(args)
                    args_preview = args_str[:200] + "..." if len(args_str) > 200 else args_str
                    arg_text = Text()
                    arg_text.append("    │     ", style="dim")
                    arg_text.append("args: ", style="dim italic")
                    arg_text.append(args_preview, style="dim")
                    console.print(arg_text)

        elif event_type == "tool.execution_complete":
            # tool.execution_complete may not have tool name, just acknowledge completion
            tool_name = getattr(event.data, "tool_name", None) or getattr(event.data, "name", None)
            if tool_name:
                text = Text()
                text.append("    │  ", style="dim")
                text.append("✓ ", style="green")
                text.append(tool_name, style="dim")
                console.print(text)

            # In full mode, try to show result preview
            if full_mode:
                result = (
                    getattr(event.data, "result", None)
                    or getattr(event.data, "output", None)
                )
                if result:
                    result_str = str(result)
                    if len(result_str) > 200:
                        result_preview = result_str[:200] + "..."
                    else:
                        result_preview = result_str
                    result_text = Text()
                    result_text.append("    │     ", style="dim")
                    result_text.append("result: ", style="dim italic")
                    result_text.append(result_preview, style="dim")
                    console.print(result_text)

        elif event_type == "assistant.reasoning":
            # Only show reasoning in full mode
            if full_mode:
                reasoning = getattr(event.data, "content", "")
                if reasoning:
                    # Truncate long reasoning for readability
                    if len(reasoning) > 150:
                        display_reasoning = reasoning[:150] + "..."
                    else:
                        display_reasoning = reasoning
                    text = Text()
                    text.append("    │  ", style="dim")
                    text.append("💭 ", style="")
                    text.append(display_reasoning.replace("\n", " "), style="italic dim")
                    console.print(text)

        elif event_type == "subagent.started":
            agent_name = getattr(event.data, "name", "unknown")
            text = Text()
            text.append("    ├─ ", style="dim")
            text.append("🤖 ", style="")
            text.append("Sub-agent: ", style="dim")
            text.append(agent_name, style="magenta bold")
            console.print(text)

        elif event_type == "subagent.completed":
            agent_name = getattr(event.data, "name", "unknown")
            text = Text()
            text.append("    │  ", style="dim")
            text.append("✓ ", style="green")
            text.append(f"Sub-agent done: {agent_name}", style="dim")
            console.print(text)

        elif event_type == "assistant.turn_start":
            # Only show processing indicator in full mode
            if full_mode:
                turn = getattr(event.data, "turn", None)
                turn_info = f" (turn {turn})" if turn else ""
                text = Text()
                text.append("    │  ", style="dim")
                text.append("⏳ ", style="yellow")
                text.append(f"Processing{turn_info}...", style="dim italic")
                console.print(text)

    async def _ensure_client_started(self) -> None:
        """Ensure the Copilot client is started."""
        if self._client is None:
            self._client = CopilotClient()
        if not self._started:
            await self._client.start()
            self._started = True

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (1-indexed).
            config: Retry configuration.

        Returns:
            Delay in seconds before next retry.
        """
        # Exponential backoff: base * 2^(attempt-1)
        delay = config.base_delay * (2 ** (attempt - 1))

        # Cap at max delay
        delay = min(delay, config.max_delay)

        # Add jitter (random fraction of delay)
        if config.jitter > 0:
            jitter_amount = delay * config.jitter * random.random()
            delay += jitter_amount

        return delay

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

        Raises:
            ProviderError: If SDK is not available or connection fails.
        """
        if self._mock_handler is not None:
            return True

        if not COPILOT_SDK_AVAILABLE:
            raise ProviderError(
                "GitHub Copilot SDK is not installed",
                suggestion="Install with: pip install github-copilot-sdk",
                is_retryable=False,
            )

        try:
            await self._ensure_client_started()
            return True
        except Exception as e:
            raise ProviderError(
                f"Failed to connect to Copilot SDK: {e}",
                suggestion=(
                    "Ensure the Copilot CLI is installed and you have an active "
                    "GitHub Copilot subscription. Install CLI: "
                    "https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli"
                ),
                is_retryable=False,
            ) from e

    async def close(self) -> None:
        """Close Copilot SDK client.

        Releases any resources held by the SDK client.
        """
        if self._client is not None and self._started:
            try:
                await self._client.stop()
            except Exception:
                pass  # Ignore errors during cleanup
        self._client = None
        self._started = False
        self._call_history.clear()
        self._retry_history.clear()

    def get_call_history(self) -> list[dict[str, Any]]:
        """Get the history of execute calls.

        This is useful for testing to verify that agents were
        called with the expected parameters.

        Returns:
            List of call records with agent_name, prompt, context, tools, model.
        """
        return self._call_history.copy()

    def get_retry_history(self) -> list[dict[str, Any]]:
        """Get the history of retry attempts.

        This is useful for testing retry behavior.

        Returns:
            List of retry records with attempt, agent_name, error, is_retryable, delay.
        """
        return self._retry_history.copy()

    def clear_call_history(self) -> None:
        """Clear the call history."""
        self._call_history.clear()

    def clear_retry_history(self) -> None:
        """Clear the retry history."""
        self._retry_history.clear()

    def set_retry_config(self, config: RetryConfig) -> None:
        """Update the retry configuration.

        Args:
            config: New retry configuration.
        """
        self._retry_config = config
