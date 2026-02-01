"""Edge case tests for ClaudeProvider.

Tests cover:
- Empty stop_sequences list
- Metadata with special characters
- max_tokens exceeding model limits
- Parameter validation edge cases
- Empty/unusual responses
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from copilot_conductor.config.schema import AgentDef, OutputField
from copilot_conductor.exceptions import ProviderError, ValidationError
from copilot_conductor.providers.claude import ClaudeProvider


class TestClaudeEdgeCases:
    """Tests for edge cases in Claude provider."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_empty_stop_sequences(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that empty stop_sequences list is handled correctly."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        # Should not raise error
        provider = ClaudeProvider(stop_sequences=[])
        assert provider._stop_sequences == []

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_metadata_with_special_characters(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test metadata with special characters."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        metadata = {
            "user_id": "user@example.com",
            "trace": "request/123/test",
            "description": 'Test with "quotes" and \'apostrophes\'',
        }

        provider = ClaudeProvider(metadata=metadata)
        assert provider._metadata == metadata

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_temperature_validation_edge_cases(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test temperature validation at boundaries."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        # Valid boundaries
        provider = ClaudeProvider(temperature=0.0)
        assert provider._default_temperature == 0.0

        provider = ClaudeProvider(temperature=1.0)
        assert provider._default_temperature == 1.0

        # Invalid - below range
        with pytest.raises(ValidationError, match="Temperature must be between 0.0 and 1.0"):
            ClaudeProvider(temperature=-0.1)

        # Invalid - above range
        with pytest.raises(ValidationError, match="Temperature must be between 0.0 and 1.0"):
            ClaudeProvider(temperature=1.1)

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_top_p_validation_edge_cases(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test top_p validation at boundaries."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        # Valid boundaries
        provider = ClaudeProvider(top_p=0.0)
        assert provider._top_p == 0.0

        provider = ClaudeProvider(top_p=1.0)
        assert provider._top_p == 1.0

        # Invalid - below range
        with pytest.raises(ValidationError, match="top_p must be between 0.0 and 1.0"):
            ClaudeProvider(top_p=-0.1)

        # Invalid - above range
        with pytest.raises(ValidationError, match="top_p must be between 0.0 and 1.0"):
            ClaudeProvider(top_p=1.1)

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_top_k_validation_edge_cases(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test top_k validation."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        # Valid
        provider = ClaudeProvider(top_k=1)
        assert provider._top_k == 1

        provider = ClaudeProvider(top_k=100)
        assert provider._top_k == 100

        # Invalid - zero
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            ClaudeProvider(top_k=0)

        # Invalid - negative
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            ClaudeProvider(top_k=-1)

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_empty_response_handling(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test handling of empty response content."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))

        # Mock response with empty content
        mock_response = Mock()
        mock_response.content = []
        mock_response.usage = Mock(input_tokens=10, output_tokens=0)
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()

        # Agent without output schema - should handle empty content
        agent = AgentDef(name="test_agent", prompt="Test prompt")
        context = {}
        rendered_prompt = "Test prompt"

        result = await provider.execute(agent, context, rendered_prompt)
        assert result.content == {"text": ""}

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_retry_history_exposure(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that retry history can be accessed for debugging."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()

        # Initially empty
        history = provider.get_retry_history()
        assert history == []
        assert isinstance(history, list)

        # Ensure it returns a copy (not the internal list)
        history.append({"test": "data"})
        assert provider.get_retry_history() == []
