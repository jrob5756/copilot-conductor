"""Tests for retry logic backoff and jitter calculation.

Verifies:
- Exponential backoff calculation
- Jitter randomization
- Retry-after header handling
- Max delay capping
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from copilot_conductor.config.schema import AgentDef
from copilot_conductor.providers.claude import ClaudeProvider, RetryConfig


class TestClaudeRetryBackoff:
    """Tests for retry backoff and jitter calculations."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_calculate_delay_exponential_backoff(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that delay calculation uses exponential backoff."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        config = RetryConfig(base_delay=1.0, max_delay=100.0, jitter=0.0)

        # Exponential backoff: base * 2^(attempt-1)
        # Attempt 1: 1.0 * 2^0 = 1.0
        delay1 = provider._calculate_delay(1, config)
        assert delay1 == 1.0

        # Attempt 2: 1.0 * 2^1 = 2.0
        delay2 = provider._calculate_delay(2, config)
        assert delay2 == 2.0

        # Attempt 3: 1.0 * 2^2 = 4.0
        delay3 = provider._calculate_delay(3, config)
        assert delay3 == 4.0

        # Attempt 4: 1.0 * 2^3 = 8.0
        delay4 = provider._calculate_delay(4, config)
        assert delay4 == 8.0

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    def test_calculate_delay_max_cap(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that delay is capped at max_delay."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        config = RetryConfig(base_delay=10.0, max_delay=30.0, jitter=0.0)

        # Attempt 1: 10.0 * 2^0 = 10.0 (below max)
        delay1 = provider._calculate_delay(1, config)
        assert delay1 == 10.0

        # Attempt 2: 10.0 * 2^1 = 20.0 (below max)
        delay2 = provider._calculate_delay(2, config)
        assert delay2 == 20.0

        # Attempt 3: 10.0 * 2^2 = 40.0 -> capped to 30.0
        delay3 = provider._calculate_delay(3, config)
        assert delay3 == 30.0

        # Attempt 4: 10.0 * 2^3 = 80.0 -> capped to 30.0
        delay4 = provider._calculate_delay(4, config)
        assert delay4 == 30.0

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @patch("copilot_conductor.providers.claude.random.random")
    def test_calculate_delay_with_jitter(
        self,
        mock_random: Mock,
        mock_anthropic_module: Mock,
        mock_anthropic_class: Mock,
    ) -> None:
        """Test that jitter is correctly added to delay."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        # Control random.random() to return predictable values
        mock_random.return_value = 0.5

        provider = ClaudeProvider()
        config = RetryConfig(base_delay=10.0, max_delay=100.0, jitter=0.25)

        # Attempt 1: base=10.0, jitter = 10.0 * 0.25 * 0.5 = 1.25
        # Total: 10.0 + 1.25 = 11.25
        delay1 = provider._calculate_delay(1, config)
        assert delay1 == 11.25

        # Attempt 2: base=20.0, jitter = 20.0 * 0.25 * 0.5 = 2.5
        # Total: 20.0 + 2.5 = 22.5
        delay2 = provider._calculate_delay(2, config)
        assert delay2 == 22.5

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @patch("copilot_conductor.providers.claude.random.random")
    def test_jitter_randomization_range(
        self,
        mock_random: Mock,
        mock_anthropic_module: Mock,
        mock_anthropic_class: Mock,
    ) -> None:
        """Test that jitter produces values within expected range."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        config = RetryConfig(base_delay=10.0, max_delay=100.0, jitter=0.25)

        # Test with minimum random value (0.0)
        mock_random.return_value = 0.0
        delay_min = provider._calculate_delay(1, config)
        assert delay_min == 10.0  # No jitter added

        # Test with maximum random value (1.0)
        mock_random.return_value = 1.0
        delay_max = provider._calculate_delay(1, config)
        # Jitter = 10.0 * 0.25 * 1.0 = 2.5
        assert delay_max == 12.5

        # Verify range: [base, base + base*jitter]
        assert 10.0 <= delay_min <= delay_max <= 12.5

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_retry_respects_retry_after_header(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that retry-after header overrides calculated delay."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))

        # Create a mock RateLimitError with retry-after header
        mock_error = Mock()
        mock_error.status_code = 429
        mock_error.response = Mock()
        mock_error.response.headers = {"retry-after": "60"}
        type(mock_error).__name__ = "RateLimitError"

        # First call fails with rate limit, second succeeds
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Success")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        mock_client.messages.create = AsyncMock(side_effect=[mock_error, mock_response])
        mock_anthropic_class.return_value = mock_client

        # Import after patching
        from copilot_conductor.providers.claude import ClaudeProvider

        provider = ClaudeProvider(retry_config=RetryConfig(max_attempts=2))

        # Execute - should retry with delay from retry-after header
        agent = AgentDef(name="test_agent", prompt="Test")
        with patch("copilot_conductor.providers.claude.asyncio.sleep") as mock_sleep:
            await provider.execute(agent, {}, "Test")

            # Verify sleep was called with retry-after value (60s), not calculated delay
            mock_sleep.assert_called_once()
            # Should be approximately 60 (from header), not exponential backoff
            assert mock_sleep.call_args[0][0] == 60.0

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_retry_history_includes_delay(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that retry history records the calculated delay."""
        mock_anthropic_module.__version__ = "0.77.0"
        mock_client = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))

        # Create a retryable error
        mock_error = Mock()
        mock_error.status_code = 503
        type(mock_error).__name__ = "APIStatusError"

        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Success")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        mock_client.messages.create = AsyncMock(side_effect=[mock_error, mock_response])
        mock_anthropic_class.return_value = mock_client

        # Import after patching
        from copilot_conductor.providers.claude import ClaudeProvider

        provider = ClaudeProvider(retry_config=RetryConfig(max_attempts=2, base_delay=2.0, jitter=0.0))

        # Execute - should retry once
        agent = AgentDef(name="test_agent", prompt="Test")
        with patch("copilot_conductor.providers.claude.asyncio.sleep"):
            await provider.execute(agent, {}, "Test")

        # Check retry history
        history = provider.get_retry_history()
        assert len(history) == 1
        assert "delay" in history[0]
        # With base_delay=2.0, jitter=0.0, first retry should be 2.0s
        assert history[0]["delay"] == 2.0
