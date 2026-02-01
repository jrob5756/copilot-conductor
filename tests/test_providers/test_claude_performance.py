"""Performance tests for Claude provider.

These tests measure performance characteristics of the Claude provider
and ensure it meets acceptable latency and throughput requirements.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from copilot_conductor.config.schema import AgentDef, OutputField
from copilot_conductor.providers.claude import ANTHROPIC_SDK_AVAILABLE, ClaudeProvider


@pytest.mark.skipif(not ANTHROPIC_SDK_AVAILABLE, reason="Anthropic SDK not installed")
@pytest.mark.performance
@pytest.mark.asyncio
async def test_provider_initialization_latency():
    """Test that provider initialization completes within acceptable time."""
    start = time.perf_counter()
    provider = ClaudeProvider()
    elapsed = time.perf_counter() - start

    # Initialization should be < 100ms (sync operation)
    assert elapsed < 0.1, f"Initialization took {elapsed:.3f}s, expected < 0.1s"

    await provider.close()


@pytest.mark.skipif(not ANTHROPIC_SDK_AVAILABLE, reason="Anthropic SDK not installed")
@pytest.mark.performance
@pytest.mark.asyncio
async def test_retry_backoff_timing():
    """Test that retry backoff timing follows exponential pattern."""
    provider = ClaudeProvider()

    # Mock client to always raise retryable error
    mock_client = AsyncMock()

    # Create a mock error that will be treated as retryable
    mock_error = Exception("Connection timeout")
    mock_error.__class__.__name__ = "APITimeoutError"

    mock_client.messages.create = AsyncMock(side_effect=mock_error)
    provider._client = mock_client

    agent = AgentDef(
        name="test",
        prompt="test",
    )

    start = time.perf_counter()

    with pytest.raises(Exception):  # ProviderError after retries
        await provider.execute(agent, {}, "test prompt")

    elapsed = time.perf_counter() - start

    # With 3 attempts and exponential backoff (1s base, 2s, 4s):
    # Total expected: ~7s (1s + 2s + 4s delays between attempts)
    # Allow some variance for execution overhead
    assert 6.5 < elapsed < 8.0, f"Retry timing {elapsed:.2f}s outside expected range 6.5-8.0s"

    # Verify retry history
    retry_history = provider.get_retry_history()
    assert len(retry_history) == 3, "Expected 3 retry attempts"

    await provider.close()


@pytest.mark.skipif(not ANTHROPIC_SDK_AVAILABLE, reason="Anthropic SDK not installed")
@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_request_handling():
    """Test that provider can handle multiple concurrent requests."""
    provider = ClaudeProvider()

    # Mock successful API responses
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            type="tool_use",
            name="emit_output",
            input={"result": "test"},
        )
    ]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    provider._client = mock_client

    agent = AgentDef(
        name="test",
        prompt="test",
        output={"result": OutputField(type="string")},
    )

    # Execute 10 concurrent requests
    start = time.perf_counter()
    tasks = [provider.execute(agent, {}, "test prompt") for _ in range(10)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    # All requests should complete
    assert len(results) == 10
    assert all(r.content == {"result": "test"} for r in results)

    # Concurrent execution should be faster than sequential
    # (Even with mocked API, asyncio overhead should be minimal)
    assert elapsed < 1.0, f"10 concurrent requests took {elapsed:.2f}s, expected < 1.0s"

    await provider.close()


@pytest.mark.skipif(not ANTHROPIC_SDK_AVAILABLE, reason="Anthropic SDK not installed")
@pytest.mark.performance
@pytest.mark.asyncio
async def test_parse_recovery_latency():
    """Test that parse recovery adds acceptable latency."""
    provider = ClaudeProvider()

    # Mock client to return text on first call, tool_use on recovery
    mock_client = AsyncMock()
    
    # First call: return invalid JSON text
    text_response = MagicMock()
    text_response.content = [MagicMock(type="text", text="not json")]
    text_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    
    # Second call (recovery): return valid tool_use
    tool_response = MagicMock()
    tool_response.content = [
        MagicMock(
            type="tool_use",
            name="emit_output",
            input={"result": "recovered"},
        )
    ]
    tool_response.usage = MagicMock(input_tokens=15, output_tokens=10)
    
    # Set up side effects: first call returns text, second returns tool
    mock_client.messages.create = AsyncMock(side_effect=[text_response, tool_response])
    provider._client = mock_client

    agent = AgentDef(
        name="test",
        prompt="test",
        output={"result": OutputField(type="string")},
    )

    start = time.perf_counter()
    result = await provider.execute(agent, {}, "test prompt")
    elapsed = time.perf_counter() - start

    # Parse recovery should complete quickly (< 500ms with mocked API)
    assert elapsed < 0.5, f"Parse recovery took {elapsed:.3f}s, expected < 0.5s"
    
    # Result should be from recovery
    assert result.content == {"result": "recovered"}

    await provider.close()


@pytest.mark.skipif(not ANTHROPIC_SDK_AVAILABLE, reason="Anthropic SDK not installed")
@pytest.mark.performance
@pytest.mark.asyncio
async def test_memory_efficiency():
    """Test that provider doesn't accumulate excessive state."""
    import sys

    provider = ClaudeProvider()

    # Record initial state size
    initial_history_size = len(provider._retry_history)

    # Simulate multiple successful executions (no retries)
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            type="tool_use",
            name="emit_output",
            input={"result": "test"},
        )
    ]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    provider._client = mock_client

    agent = AgentDef(
        name="test",
        prompt="test",
        output={"result": OutputField(type="string")},
    )

    # Execute 100 times
    for _ in range(100):
        await provider.execute(agent, {}, "test prompt")

    # Retry history should not grow (no retries occurred)
    final_history_size = len(provider._retry_history)
    assert final_history_size == initial_history_size, "Retry history leaked on successful calls"

    # Provider object size should be reasonable (< 1MB)
    provider_size = sys.getsizeof(provider)
    assert provider_size < 1_000_000, f"Provider size {provider_size} bytes exceeds 1MB"

    await provider.close()
