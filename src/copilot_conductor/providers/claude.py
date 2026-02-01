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
    from anthropic import AsyncAnthropic

    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
    AsyncAnthropic = None  # type: ignore[misc, assignment]
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

        self._client: AsyncAnthropic | None = None
        self._api_key = api_key
        self._default_model = model or "claude-3-5-sonnet-latest"
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens or 8192
        self._timeout = timeout
        self._sdk_version: str | None = None
        self._max_parse_recovery_attempts = 2  # Max retry attempts for malformed JSON
        self._max_schema_depth = 10  # Max nesting depth for recursive schema building

        # Initialize the client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Anthropic client and verify SDK version."""
        if not ANTHROPIC_SDK_AVAILABLE or AsyncAnthropic is None:
            return

        self._client = AsyncAnthropic(
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

    async def _verify_available_models(self) -> None:
        """List and log available models, warn if default model is unavailable.

        Note: This is async and should be called from async context (e.g., validate_connection).
        """
        if self._client is None:
            return

        try:
            # Call client.models.list() to get available models (async)
            models_page = await self._client.models.list()
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
            await self._client.models.list()
            # Also verify available models on first connection
            await self._verify_available_models()
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    async def close(self) -> None:
        """Release provider resources and close connections."""
        if self._client is not None:
            # AsyncAnthropic uses httpx AsyncClient internally which should be closed
            await self._client.close()
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
            # Execute non-streaming API call with parse recovery
            response = await self._execute_with_parse_recovery(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                sdk_tools=sdk_tools,
                output_schema=agent.output,
            )

            # Extract structured output
            content = self._extract_output(response, agent.output)

            # Validate output if schema is defined
            if agent.output:
                validate_output(content, agent.output)

            # Extract token usage using dedicated method
            tokens_used = self._extract_token_usage(response)

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
                                "Temperature must be between 0.0 and 1.0 (enforced by Claude SDK)"
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

    async def _execute_api_call(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
        max_tokens: int,
        sdk_tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Execute non-streaming Claude API call using AsyncAnthropic.

        This method makes an asynchronous (non-streaming) call to the Claude
        messages.create() API endpoint. It does not handle streaming responses.

        Args:
            messages: Message history to send.
            model: Model identifier.
            temperature: Temperature setting (0.0-1.0, enforced by SDK).
            max_tokens: Maximum output tokens.
            sdk_tools: Optional tool definitions for structured output.

        Returns:
            Claude API response object with content blocks and usage metadata.

        Raises:
            ProviderError: If client not initialized or API call fails.

        Note:
            This is a non-streaming implementation. Streaming support is
            deferred to Phase 2+ of the Claude SDK integration.
        """
        if self._client is None:
            raise ProviderError("Claude client not initialized")

        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        if sdk_tools:
            kwargs["tools"] = sdk_tools

        # Execute non-streaming API call (async)
        logger.debug(
            f"Executing non-streaming Claude API call: model={model}, "
            f"max_tokens={max_tokens}, timeout={self._timeout}s"
        )
        response = await self._client.messages.create(**kwargs)

        return response

    async def _execute_with_parse_recovery(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
        max_tokens: int,
        sdk_tools: list[dict[str, Any]] | None,
        output_schema: dict[str, OutputField] | None,
    ) -> Any:
        """Execute API call with parse recovery for malformed JSON responses.

        This method handles the fallback case where Claude returns text instead
        of using the tool, and the text contains malformed JSON. It will retry
        up to max_parse_recovery_attempts times with clarifying prompts.

        Args:
            messages: Message history to send.
            model: Model identifier.
            temperature: Temperature setting.
            max_tokens: Maximum output tokens.
            sdk_tools: Tool definitions for structured output.
            output_schema: Expected output schema (None if no schema).

        Returns:
            Claude API response.

        Raises:
            ProviderError: If all retry attempts fail with context about attempts.
        """
        # Track recovery attempts for error reporting
        recovery_history: list[str] = []

        # Initial attempt using non-streaming API call
        response = await self._execute_api_call(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            sdk_tools=sdk_tools,
        )

        # If no output schema, return immediately (no recovery needed)
        if not output_schema:
            return response

        # Check if we got tool_use (success path)
        if self._extract_structured_output(response) is not None:
            logger.debug("Extracted structured output from tool_use block")
            return response

        # Check if we can extract JSON from text (fallback success path)
        json_content = self._extract_json_fallback(response)
        if json_content is not None:
            logger.info("Claude returned text instead of tool_use, but JSON extraction succeeded")
            return response

        # Parse recovery: JSON extraction failed
        initial_text = self._extract_text_from_response(response)
        failure_reason = self._diagnose_json_failure(initial_text)
        recovery_history.append(f"Attempt 0 (initial): {failure_reason}")
        logger.info(f"Initial extraction failed: {failure_reason}, starting parse recovery")

        for attempt in range(1, self._max_parse_recovery_attempts + 1):
            logger.info(f"Parse recovery attempt {attempt}/{self._max_parse_recovery_attempts}")

            # Append recovery message with specific error context
            recovery_messages = messages.copy()
            recovery_messages.append(
                {
                    "role": "assistant",
                    "content": initial_text,
                }
            )
            recovery_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response did not contain valid JSON. {failure_reason} "
                        "Please provide your response in valid JSON format.\n\n"
                        "IMPORTANT: Use the 'emit_output' tool to return your response "
                        "in the required structured format."
                    ),
                }
            )

            # Retry API call using non-streaming method
            response = await self._execute_api_call(
                messages=recovery_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                sdk_tools=sdk_tools,
            )

            # Check if recovery succeeded (tool_use)
            if self._extract_structured_output(response) is not None:
                logger.info(f"Parse recovery succeeded on attempt {attempt} (tool_use)")
                return response

            # Check if recovery succeeded (JSON fallback)
            json_content = self._extract_json_fallback(response)
            if json_content is not None:
                logger.info(f"Parse recovery succeeded on attempt {attempt} (JSON)")
                return response

            # Record failure for this attempt
            attempt_text = self._extract_text_from_response(response)
            attempt_failure = self._diagnose_json_failure(attempt_text)
            recovery_history.append(f"Attempt {attempt}: {attempt_failure}")
            # Update for next iteration
            initial_text = attempt_text
            failure_reason = attempt_failure

        # All recovery attempts exhausted - raise detailed error
        logger.error(f"Parse recovery exhausted after {self._max_parse_recovery_attempts} attempts")
        raise ProviderError(
            f"Failed to extract valid JSON after {self._max_parse_recovery_attempts} "
            "recovery attempts",
            suggestion=(
                "Claude did not use the emit_output tool and returned invalid JSON. "
                f"Recovery history: {'; '.join(recovery_history)}"
            ),
        )

    def _diagnose_json_failure(self, text: str) -> str:
        """Diagnose why JSON extraction failed from text response.

        Args:
            text: The text content that failed to parse.

        Returns:
            Human-readable diagnosis of the failure.
        """
        if not text.strip():
            return "Response was empty."

        # Check for incomplete JSON patterns
        if "{" in text and "}" not in text:
            return "Found incomplete JSON (opening brace without closing)."
        if "[" in text and "]" not in text:
            return "Found incomplete JSON (opening bracket without closing)."

        # Check if it looks like JSON but has syntax errors
        if text.strip().startswith(("{", "[")):
            return "Found malformed JSON (syntax error in structure)."

        # Try to find JSON code block
        if "```" in text:
            if "```json" not in text:
                return "Found code block but not marked as JSON."
            return "Found JSON code block but it contains syntax errors."

        # No JSON-like content found
        return "No JSON content found in response text."

    def _process_response_content_blocks(
        self, response: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Process content blocks from Claude response.

        Extracts both text content and tool_use content from the response,
        categorizing each block by its type.

        Note:
            This method is provided for debugging and future features (e.g.,
            detailed response logging, tool call tracing). It is not currently
            used in the main execution flow but is tested to ensure correctness
            for when it's needed.

        Args:
            response: Claude API response with content blocks.

        Returns:
            Tuple of (all_blocks, tool_use_data) where:
                - all_blocks: List of dicts describing each content block
                - tool_use_data: Dict from emit_output tool_use, or None

        Note:
            This method processes non-streaming responses only. Each response
            contains a list of content blocks which can be text or tool_use.
        """
        blocks = []
        tool_use_data = None

        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    blocks.append(
                        {
                            "type": "text",
                            "text": block.text,
                        }
                    )
                elif block.type == "tool_use":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "name": block.name,
                            "id": getattr(block, "id", None),
                        }
                    )
                    # Capture emit_output tool data
                    if block.name == "emit_output":
                        tool_use_data = dict(block.input)

        logger.debug(f"Processed {len(blocks)} content blocks from response")
        return blocks, tool_use_data

    def _extract_token_usage(self, response: Any) -> int | None:
        """Extract token usage from Claude response.

        Args:
            response: Claude API response with usage metadata.

        Returns:
            Total tokens used (input + output), or None if not available.

        Note:
            Claude response.usage contains input_tokens and output_tokens.
            This method sums both to provide total usage.
        """
        if not hasattr(response, "usage"):
            logger.debug("Response does not contain usage metadata")
            return None

        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        total = input_tokens + output_tokens

        logger.debug(f"Token usage: {input_tokens} input + {output_tokens} output = {total} total")
        return total

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
        properties = self._build_json_schema_properties(output_schema)
        required = list(output_schema.keys())

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

    def _build_json_schema_properties(
        self, schema: dict[str, OutputField], depth: int = 0
    ) -> dict[str, Any]:
        """Build JSON Schema properties from OutputField definitions.

        Recursively handles nested objects and arrays with depth limiting.

        Args:
            schema: Dictionary mapping field names to OutputField definitions.
            depth: Current nesting depth (for recursion safety).

        Returns:
            Dictionary of JSON Schema property definitions.

        Raises:
            ValidationError: If schema nesting exceeds max depth.
        """
        if depth > self._max_schema_depth:
            raise ValidationError(
                f"Schema nesting depth exceeds maximum of {self._max_schema_depth} levels",
                suggestion="Simplify your output schema to reduce nesting depth",
            )

        properties: dict[str, Any] = {}

        for field_name, field_def in schema.items():
            prop: dict[str, Any] = {
                "type": self._map_type_to_json_schema(field_def.type),
            }

            if field_def.description:
                prop["description"] = field_def.description

            # Handle nested object schemas
            if field_def.type == "object" and field_def.properties:
                prop["properties"] = self._build_json_schema_properties(
                    field_def.properties, depth=depth + 1
                )
                # All properties in OutputField schemas are required
                # (OutputField has no 'required' attribute, all fields are mandatory)
                prop["required"] = list(field_def.properties.keys())

            # Handle array schemas with item definitions
            if field_def.type == "array" and field_def.items:
                items_schema = self._build_single_field_schema(field_def.items, depth=depth + 1)
                prop["items"] = items_schema

            properties[field_name] = prop

        return properties

    def _build_single_field_schema(self, field: OutputField, depth: int = 0) -> dict[str, Any]:
        """Build JSON Schema for a single field (used for array items).

        Args:
            field: The OutputField definition.
            depth: Current nesting depth (for recursion safety).

        Returns:
            JSON Schema definition for the field.

        Raises:
            ValidationError: If schema nesting exceeds max depth.
        """
        if depth > self._max_schema_depth:
            raise ValidationError(
                f"Schema nesting depth exceeds maximum of {self._max_schema_depth} levels",
                suggestion="Simplify your output schema to reduce nesting depth",
            )

        schema: dict[str, Any] = {
            "type": self._map_type_to_json_schema(field.type),
        }

        if field.description:
            schema["description"] = field.description

        # Handle nested objects in array items
        if field.type == "object" and field.properties:
            schema["properties"] = self._build_json_schema_properties(
                field.properties, depth=depth + 1
            )
            # All properties are required
            schema["required"] = list(field.properties.keys())

        # Handle nested arrays (array of arrays)
        if field.type == "array" and field.items:
            schema["items"] = self._build_single_field_schema(field.items, depth=depth + 1)

        return schema

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

    def _extract_text_from_response(self, response: Any) -> str:
        """Extract raw text content from Claude response.

        Used for building message history during parse recovery.

        Args:
            response: Claude API response.

        Returns:
            Combined text content from all text blocks.
        """
        text_parts = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)
