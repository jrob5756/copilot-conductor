"""Anthropic Claude SDK provider implementation.

This module provides the ClaudeProvider class for executing agents
using the Anthropic Claude SDK with tool-based structured output.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from copilot_conductor.exceptions import ProviderError, ValidationError
from copilot_conductor.executor.output import validate_output
from copilot_conductor.providers.base import AgentOutput, AgentProvider

if TYPE_CHECKING:
    from copilot_conductor.config.schema import AgentDef, OutputField

# Try to import the Anthropic SDK
try:
    import anthropic
    from anthropic import Anthropic

    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
    Anthropic = None  # type: ignore[misc, assignment]
    anthropic = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ClaudeProvider(AgentProvider):
    """Anthropic Claude SDK provider.

    Translates Conductor agent definitions into Claude SDK calls and
    normalizes responses into AgentOutput format. Uses tool-based
    structured output extraction for reliable JSON responses.

    Supports non-streaming message execution with error handling,
    retry logic, and temperature validation.

    Example:
        >>> provider = ClaudeProvider(api_key="sk-...")
        >>> await provider.validate_connection()
        True
        >>> await provider.close()
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 600.0,
    ) -> None:
        """Initialize the Claude provider.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Default model to use. Defaults to "claude-3-5-sonnet-latest".
                This default is chosen for stability and to avoid dated model
                deprecation risk. The "-latest" suffix ensures compatibility
                with model updates without requiring configuration changes.
            temperature: Default temperature (0.0-1.0). SDK enforces range.
            max_tokens: Maximum output tokens. Defaults to 8192.
            timeout: Request timeout in seconds. Defaults to 600s.

        Raises:
            ProviderError: If SDK is not installed.
        """
        if not ANTHROPIC_SDK_AVAILABLE:
            raise ProviderError(
                "Anthropic SDK not installed",
                suggestion="Install with: uv add 'anthropic>=0.77.0,<1.0.0'",
            )

        self._client: Anthropic | None = None
        self._api_key = api_key
        self._default_model = model or "claude-3-5-sonnet-latest"
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens or 8192
        self._timeout = timeout
        self._sdk_version: str | None = None

        # Initialize the client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Anthropic client and verify SDK version."""
        if not ANTHROPIC_SDK_AVAILABLE or Anthropic is None:
            return

        self._client = Anthropic(
            api_key=self._api_key,
            timeout=self._timeout,
        )

        # Log SDK version
        if anthropic is not None:
            self._sdk_version = getattr(anthropic, "__version__", "unknown")
            logger.info(f"Initialized Claude provider with SDK version {self._sdk_version}")

            # Warn if version is outside expected range
            if self._sdk_version != "unknown":
                try:
                    major, minor, patch = self._sdk_version.split(".")
                    version_parts = (int(major), int(minor))
                    if version_parts[0] == 0 and version_parts[1] < 77:
                        logger.warning(
                            f"Anthropic SDK version {self._sdk_version} is older than 0.77.0. "
                            "Some features may not work correctly."
                        )
                    elif version_parts[0] >= 1:
                        logger.warning(
                            f"Anthropic SDK version {self._sdk_version} is >= 1.0.0. "
                            "This provider was tested with 0.77.x. Compatibility issues may occur."
                        )
                except (ValueError, AttributeError):
                    logger.debug(f"Could not parse SDK version: {self._sdk_version}")

        # List available models
        self._verify_available_models()

    def _verify_available_models(self) -> None:
        """List and log available models, warn if default model is unavailable."""
        if self._client is None:
            return

        try:
            # Call client.models.list() to get available models
            models_page = self._client.models.list()
            available_models = [model.id for model in models_page.data]

            logger.debug(f"Available Claude models: {', '.join(available_models)}")

            # Warn if default model not in list
            if self._default_model not in available_models:
                logger.warning(
                    f"Requested model '{self._default_model}' is not in the list of "
                    f"available models. API calls may fail. Available: {available_models}"
                )
        except Exception as e:
            logger.debug(f"Could not list available models: {e}")

    async def validate_connection(self) -> bool:
        """Verify the provider can connect to the Claude API.

        Returns:
            True if connection successful, False otherwise.
        """
        if self._client is None:
            return False

        try:
            # Simple test: list models to verify API key works
            self._client.models.list()
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    async def close(self) -> None:
        """Release provider resources and close connections."""
        if self._client is not None:
            # Anthropic client doesn't require explicit cleanup
            # but we can clear the reference
            self._client = None
            logger.debug("Claude provider closed")

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        tools: list[str] | None = None,
    ) -> AgentOutput:
        """Execute an agent using the Claude SDK.

        Args:
            agent: Agent definition from workflow config.
            context: Accumulated workflow context.
            rendered_prompt: Jinja2-rendered user prompt.
            tools: List of tool names available to this agent (currently unused).

        Returns:
            Normalized AgentOutput with structured content.

        Raises:
            ProviderError: If SDK execution fails.
            ValidationError: If output doesn't match schema.
        """
        if self._client is None:
            raise ProviderError("Claude client not initialized")

        # Build messages
        messages = self._build_messages(rendered_prompt)

        # Get model and parameters
        model = agent.model or self._default_model
        temperature = self._default_temperature
        max_tokens = self._default_max_tokens

        # Validate max_tokens against model-specific limits
        if "haiku" in model.lower():
            if max_tokens > 4096:
                logger.warning(
                    f"max_tokens={max_tokens} exceeds Haiku model limit of 4096. "
                    "API may reject request."
                )
        elif max_tokens > 8192:
            logger.warning(
                f"max_tokens={max_tokens} exceeds Sonnet/Opus model limit of 8192. "
                "API may reject request."
            )

        # Build tools for structured output if schema is defined
        sdk_tools = None
        if agent.output:
            sdk_tools = self._build_tools_for_structured_output(agent.output)
            # Append instruction to use the tool
            messages[-1]["content"] += (
                "\n\nPlease use the 'emit_output' tool to return your response "
                "in the required structured format."
            )

        try:
            # Execute non-streaming API call
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            if temperature is not None:
                kwargs["temperature"] = temperature

            if sdk_tools:
                kwargs["tools"] = sdk_tools

            response = self._client.messages.create(**kwargs)

            # Extract structured output
            content = self._extract_output(response, agent.output)

            # Validate output if schema is defined
            if agent.output:
                validate_output(content, agent.output)

            # Extract token usage
            tokens_used = None
            if hasattr(response, "usage"):
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return AgentOutput(
                content=content,
                raw_response=response,
                tokens_used=tokens_used,
                model=model,
            )

        except ValidationError:
            # Re-raise ValidationError without wrapping
            raise
        except Exception as e:
            # Check if it's a BadRequestError from temperature validation
            # We check both hasattr and isinstance to handle mocking correctly
            if anthropic is not None:
                try:
                    has_attr = hasattr(anthropic, "BadRequestError")
                    is_bad_request = has_attr and isinstance(e, anthropic.BadRequestError)
                    if is_bad_request and "temperature" in str(e).lower():
                        raise ValidationError(
                            f"Temperature validation failed: {e}",
                            suggestion=(
                                "Temperature must be between 0.0 and 1.0 "
                                "(enforced by Claude SDK)"
                            ),
                        ) from e
                except TypeError:
                    # isinstance can fail if BadRequestError is not a proper type (e.g., in tests)
                    pass

            # Wrap other errors as ProviderError
            raise ProviderError(
                f"Claude API call failed: {e}",
                suggestion="Check API key, model name, and request parameters",
                is_retryable=True,
            ) from e

    def _build_messages(self, rendered_prompt: str) -> list[dict[str, str]]:
        """Build message list for Claude API.

        Args:
            rendered_prompt: The user prompt to send.

        Returns:
            List of message dicts with role and content.
        """
        return [
            {
                "role": "user",
                "content": rendered_prompt,
            }
        ]

    def _build_tools_for_structured_output(
        self, output_schema: dict[str, OutputField]
    ) -> list[dict[str, Any]]:
        """Convert output schema to Claude tool definition.

        Args:
            output_schema: Agent's output schema.

        Returns:
            List containing single tool definition for structured output.
        """
        # Build JSON schema from OutputField definitions
        properties: dict[str, Any] = {}
        required: list[str] = []

        for field_name, field_def in output_schema.items():
            properties[field_name] = {
                "type": self._map_type_to_json_schema(field_def.type),
            }
            if field_def.description:
                properties[field_name]["description"] = field_def.description

            # All fields are required by default in our schema
            required.append(field_name)

        return [
            {
                "name": "emit_output",
                "description": "Emit the structured output for this task",
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        ]

    def _map_type_to_json_schema(self, field_type: str) -> str:
        """Map OutputField type to JSON Schema type.

        Args:
            field_type: The OutputField type string.

        Returns:
            Corresponding JSON Schema type.
        """
        type_mapping = {
            "string": "string",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }
        return type_mapping.get(field_type, "string")

    def _extract_output(
        self, response: Any, output_schema: dict[str, OutputField] | None
    ) -> dict[str, Any]:
        """Extract structured output from Claude response.

        Tries tool_use blocks first, falls back to text parsing.

        Args:
            response: Claude API response.
            output_schema: Expected output schema (None if no schema).

        Returns:
            Extracted content as dict.

        Raises:
            ProviderError: If extraction fails.
        """
        # If no schema, extract text content
        if not output_schema:
            return self._extract_text_content(response)

        # Try to extract from tool_use blocks
        content = self._extract_structured_output(response)
        if content is not None:
            return content

        # Fallback: try to parse JSON from text
        content = self._extract_json_fallback(response)
        if content is not None:
            return content

        # If both failed, raise error
        raise ProviderError(
            "Failed to extract structured output from Claude response",
            suggestion="Ensure the agent is using the emit_output tool or returning valid JSON",
        )

    def _extract_text_content(self, response: Any) -> dict[str, Any]:
        """Extract plain text content when no schema is defined.

        Args:
            response: Claude API response.

        Returns:
            Dict with 'text' key containing the response text.
        """
        text_parts = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)

        return {"text": "\n".join(text_parts)}

    def _extract_structured_output(self, response: Any) -> dict[str, Any] | None:
        """Extract structured output from tool_use content blocks.

        Args:
            response: Claude API response.

        Returns:
            Extracted content dict, or None if no tool_use found.
        """
        for block in response.content:
            is_tool_use = hasattr(block, "type") and block.type == "tool_use"
            if is_tool_use and block.name == "emit_output":
                return dict(block.input)
        return None

    def _extract_json_fallback(self, response: Any) -> dict[str, Any] | None:
        """Fallback: parse JSON from text content.

        Args:
            response: Claude API response.

        Returns:
            Parsed JSON dict, or None if parsing fails.
        """
        text_parts = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)

        text = "\n".join(text_parts)

        # Try to find and parse JSON
        try:
            # Look for JSON code blocks
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                json_str = text[start:end].strip()
                return json.loads(json_str)

            # Try parsing the whole text
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
