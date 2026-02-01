"""Unit tests for the ClaudeProvider implementation.

Tests cover:
- Provider initialization with SDK version verification
- Connection validation
- Basic message execution
- Structured output extraction (tool-based and fallback)
- Temperature validation (SDK-enforced behavior)
- Error handling and wrapping
"""

from unittest.mock import Mock, patch

import pytest

from copilot_conductor.config.schema import AgentDef, OutputField
from copilot_conductor.exceptions import ProviderError, ValidationError
from copilot_conductor.providers.claude import ClaudeProvider


class TestClaudeProviderInitialization:
    """Tests for ClaudeProvider initialization."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", False)
    def test_init_raises_when_sdk_not_installed(self) -> None:
        """Test that initialization raises ProviderError when SDK not available."""
        with pytest.raises(ProviderError, match="Anthropic SDK not installed"):
            ClaudeProvider()

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_init_with_default_parameters(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test initialization with default parameters."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()

        assert provider._default_model == "claude-3-5-sonnet-latest"
        assert provider._default_max_tokens == 8192
        assert provider._timeout == 600.0
        assert provider._sdk_version == "0.77.0"
        mock_anthropic_class.assert_called_once()

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_init_with_custom_parameters(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test initialization with custom parameters."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider(
            api_key="test-key",
            model="claude-3-opus-20240229",
            temperature=0.5,
            max_tokens=4096,
            timeout=300.0,
        )

        assert provider._api_key == "test-key"
        assert provider._default_model == "claude-3-opus-20240229"
        assert provider._default_temperature == 0.5
        assert provider._default_max_tokens == 4096
        assert provider._timeout == 300.0

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @patch("copilot_conductor.providers.claude.logger")
    def test_sdk_version_warning_old_version(
        self,
        mock_logger: Mock,
        mock_anthropic_module: Mock,
        mock_anthropic_class: Mock,
    ) -> None:
        """Test warning when SDK version is older than 0.77.0."""
        mock_anthropic_module.__version__ = "0.76.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_anthropic_class.return_value = mock_client

        ClaudeProvider()

        # Check that SDK version warning was issued (may be multiple warnings)
        assert mock_logger.warning.called
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("0.76.0" in call and "older than 0.77.0" in call for call in warning_calls)

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @patch("copilot_conductor.providers.claude.logger")
    def test_sdk_version_warning_future_version(
        self,
        mock_logger: Mock,
        mock_anthropic_module: Mock,
        mock_anthropic_class: Mock,
    ) -> None:
        """Test warning when SDK version is >= 1.0.0."""
        mock_anthropic_module.__version__ = "1.0.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_anthropic_class.return_value = mock_client

        ClaudeProvider()

        # Check that SDK version warning was issued (may be multiple warnings)
        assert mock_logger.warning.called
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("1.0.0" in call and ">= 1.0.0" in call for call in warning_calls)


class TestModelVerification:
    """Tests for model availability verification."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @patch("copilot_conductor.providers.claude.logger")
    def test_model_verification_lists_available_models(
        self,
        mock_logger: Mock,
        mock_anthropic_module: Mock,
        mock_anthropic_class: Mock,
    ) -> None:
        """Test that available models are listed and logged."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()

        # Mock models.list() response
        mock_model1 = Mock()
        mock_model1.id = "claude-3-5-sonnet-latest"
        mock_model2 = Mock()
        mock_model2.id = "claude-3-opus-20240229"
        mock_client.models.list.return_value = Mock(data=[mock_model1, mock_model2])
        mock_anthropic_class.return_value = mock_client

        ClaudeProvider()

        # Check that models were listed
        mock_client.models.list.assert_called_once()

        # Check that available models were logged (at DEBUG level)
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Available Claude models" in call for call in debug_calls)

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @patch("copilot_conductor.providers.claude.logger")
    def test_model_verification_warns_unavailable_model(
        self,
        mock_logger: Mock,
        mock_anthropic_module: Mock,
        mock_anthropic_class: Mock,
    ) -> None:
        """Test warning when requested model is not available."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()

        # Mock models.list() with different models
        mock_model = Mock()
        mock_model.id = "claude-3-opus-20240229"
        mock_client.models.list.return_value = Mock(data=[mock_model])
        mock_anthropic_class.return_value = mock_client

        # Request a model that's not in the list
        ClaudeProvider(model="claude-sonnet-4-20250514")

        # Check warning was logged
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("not in the list of available models" in call for call in warning_calls)


class TestConnectionValidation:
    """Tests for connection validation."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_validate_connection_success(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test successful connection validation."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        result = await provider.validate_connection()

        assert result is True

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_validate_connection_failure(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test connection validation failure."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("API key invalid")
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        result = await provider.validate_connection()

        assert result is False


class TestCloseMethod:
    """Tests for resource cleanup."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_close_clears_client(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that close() clears the client reference."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        assert provider._client is not None

        await provider.close()
        assert provider._client is None


class TestBasicExecution:
    """Tests for basic message execution without structured output."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_execute_simple_message(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test executing a simple message without output schema."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])

        # Mock response
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello, world!"

        mock_response = Mock()
        mock_response.content = [mock_text_block]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        agent = AgentDef(name="test", prompt="Say hello")

        result = await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Say hello",
        )

        assert result.content == {"text": "Hello, world!"}
        assert result.tokens_used == 15
        assert result.model == "claude-3-5-sonnet-latest"

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_execute_with_agent_model(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that agent model overrides default."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])

        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = Mock()
        mock_response.content = [mock_text_block]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        agent = AgentDef(
            name="test",
            prompt="Test",
            model="claude-3-opus-20240229",
        )

        result = await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Test",
        )

        assert result.model == "claude-3-opus-20240229"

        # Verify API was called with correct model
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-opus-20240229"

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_execute_with_temperature(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that temperature is passed to API."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])

        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = Mock()
        mock_response.content = [mock_text_block]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider(temperature=0.7)
        agent = AgentDef(name="test", prompt="Test")

        await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Test",
        )

        # Verify API was called with provider temperature
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7


class TestStructuredOutput:
    """Tests for structured output extraction using tools.

    The ClaudeProvider uses a tool-based approach where the output schema
    is converted to a tool definition that the model must use to return
    structured data.
    """

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_execute_with_structured_output_via_tool(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test structured output extraction from tool_use blocks."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])

        # Mock tool_use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "emit_output"
        mock_tool_block.input = {"answer": "42", "confidence": 0.95}

        mock_response = Mock()
        mock_response.content = [mock_tool_block]
        mock_response.usage = Mock(input_tokens=20, output_tokens=10)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        agent = AgentDef(
            name="test",
            prompt="Answer question",
            output={
                "answer": OutputField(type="string", description="The answer"),
                "confidence": OutputField(type="number", description="Confidence score"),
            },
        )

        result = await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="What is the answer?",
        )

        assert result.content == {"answer": "42", "confidence": 0.95}

        # Verify tool was included in API call
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "emit_output"

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_execute_with_json_fallback(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test fallback JSON extraction when model returns text instead of tool_use."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])

        # Mock text response with JSON
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = '```json\n{"answer": "Paris", "country": "France"}\n```'

        mock_response = Mock()
        mock_response.content = [mock_text_block]
        mock_response.usage = Mock(input_tokens=20, output_tokens=15)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        agent = AgentDef(
            name="test",
            prompt="Answer",
            output={
                "answer": OutputField(type="string"),
                "country": OutputField(type="string"),
            },
        )

        result = await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="What is the capital of France?",
        )

        assert result.content == {"answer": "Paris", "country": "France"}


class TestTemperatureValidation:
    """Tests for temperature validation behavior.

    Note: The ClaudeProvider does NOT perform its own temperature validation.
    Instead, it relies on the SDK to enforce the [0.0, 1.0] range and raises
    BadRequestError for violations. The provider catches this error and wraps
    it as a ValidationError with a clear message.

    These tests document the SDK-enforced behavior rather than testing
    provider-side validation logic.
    """

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_temperature_above_1_0_raises_validation_error(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that SDK raises BadRequestError for temperature > 1.0.

        This test verifies that the provider properly wraps the SDK's
        BadRequestError as a ValidationError with a helpful message.
        """
        mock_anthropic_module.__version__ = "0.77.0"

        # Create mock BadRequestError
        mock_bad_request_error = type(
            "BadRequestError",
            (Exception,),
            {},
        )("temperature must be between 0.0 and 1.0")
        mock_anthropic_module.BadRequestError = mock_bad_request_error.__class__

        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_client.messages.create.side_effect = mock_bad_request_error
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider(temperature=1.5)  # Invalid: > 1.0
        agent = AgentDef(
            name="test",
            prompt="Test",
        )

        with pytest.raises(ValidationError) as exc_info:
            await provider.execute(
                agent=agent,
                context={},
                rendered_prompt="Test",
            )

        assert "Temperature validation failed" in str(exc_info.value)
        assert "between 0.0 and 1.0" in str(exc_info.value)


class TestErrorHandling:
    """Tests for error handling and wrapping."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_api_error_wrapped_as_provider_error(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that API errors are wrapped as ProviderError."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_client.messages.create.side_effect = Exception("API error")
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        agent = AgentDef(name="test", prompt="Test")

        with pytest.raises(ProviderError) as exc_info:
            await provider.execute(
                agent=agent,
                context={},
                rendered_prompt="Test",
            )

        assert "Claude API call failed" in str(exc_info.value)
        assert exc_info.value.is_retryable is True

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_execute_with_no_client_raises_error(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that execute raises error if client not initialized."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        provider._client = None  # Simulate uninitialized client

        agent = AgentDef(name="test", prompt="Test")

        with pytest.raises(ProviderError, match="client not initialized"):
            await provider.execute(
                agent=agent,
                context={},
                rendered_prompt="Test",
            )

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_validation_error_for_missing_output_fields(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that missing output fields raise ValidationError."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])

        # Mock tool_use with incomplete output
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "emit_output"
        mock_tool_block.input = {"answer": "42"}  # Missing 'confidence'

        mock_response = Mock()
        mock_response.content = [mock_tool_block]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        agent = AgentDef(
            name="test",
            prompt="Test",
            output={
                "answer": OutputField(type="string"),
                "confidence": OutputField(type="number"),  # Required but missing
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            await provider.execute(
                agent=agent,
                context={},
                rendered_prompt="Test",
            )

        assert "Missing required output field: confidence" in str(exc_info.value)


class TestToolSchemaGeneration:
    """Tests for tool schema generation from output schemas."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_build_tools_for_simple_schema(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test tool generation from simple output schema."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()

        schema = {
            "result": OutputField(type="string", description="The result"),
            "score": OutputField(type="number", description="A score"),
        }

        tools = provider._build_tools_for_structured_output(schema)

        assert len(tools) == 1
        assert tools[0]["name"] == "emit_output"
        assert "input_schema" in tools[0]
        assert tools[0]["input_schema"]["type"] == "object"
        assert "result" in tools[0]["input_schema"]["properties"]
        assert "score" in tools[0]["input_schema"]["properties"]
        assert tools[0]["input_schema"]["properties"]["result"]["type"] == "string"
        assert tools[0]["input_schema"]["properties"]["score"]["type"] == "number"
        assert set(tools[0]["input_schema"]["required"]) == {"result", "score"}


class TestConcurrentExecution:
    """Tests for concurrent execution scenarios."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_concurrent_execute_calls(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that multiple concurrent execute() calls work correctly."""
        import asyncio

        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])

        # Mock responses for different calls
        call_count = 0

        def create_response():
            nonlocal call_count
            call_count += 1
            mock_text_block = Mock()
            mock_text_block.type = "text"
            mock_text_block.text = f"Response {call_count}"
            mock_response = Mock()
            mock_response.content = [mock_text_block]
            mock_response.usage = Mock(input_tokens=10, output_tokens=5)
            return mock_response

        mock_client.messages.create.side_effect = lambda **kwargs: create_response()
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        agent1 = AgentDef(name="test1", prompt="Hello 1")
        agent2 = AgentDef(name="test2", prompt="Hello 2")
        agent3 = AgentDef(name="test3", prompt="Hello 3")

        # Execute three agents concurrently
        results = await asyncio.gather(
            provider.execute(agent=agent1, context={}, rendered_prompt="Hello 1"),
            provider.execute(agent=agent2, context={}, rendered_prompt="Hello 2"),
            provider.execute(agent=agent3, context={}, rendered_prompt="Hello 3"),
        )

        # Verify all three executed successfully
        assert len(results) == 3
        assert all(result.content for result in results)
        assert all(result.tokens_used == 15 for result in results)

        # Verify all three API calls were made
        assert mock_client.messages.create.call_count == 3


class TestTextContentExtraction:
    """Tests for text content extraction with multiple blocks."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.Anthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_extract_text_content_multiple_blocks(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test extraction with multiple text blocks in response."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])

        # Mock response with multiple text blocks
        mock_text_block1 = Mock()
        mock_text_block1.type = "text"
        mock_text_block1.text = "First part. "

        mock_text_block2 = Mock()
        mock_text_block2.type = "text"
        mock_text_block2.text = "Second part."

        mock_response = Mock()
        mock_response.content = [mock_text_block1, mock_text_block2]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        agent = AgentDef(name="test", prompt="Say hello")

        result = await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="Say hello",
        )

        # Verify both text blocks are combined with newline separator
        assert result.content == {"text": "First part. \nSecond part."}

