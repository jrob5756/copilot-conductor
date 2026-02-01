"""Tests for error handling of Claude-specific configuration parameters.

Addresses review feedback about missing error handling tests for new Claude parameters.
Tests validation of temperature, max_tokens, top_p, top_k ranges and types.
"""

import pytest
from pydantic import ValidationError

from copilot_conductor.config.schema import RuntimeConfig


def test_temperature_out_of_range_low():
    """Verify temperature < 0 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            temperature=-0.1
        )
    
    errors = exc_info.value.errors()
    assert any("temperature" in str(e.get("loc", [])) for e in errors)


def test_temperature_out_of_range_high():
    """Verify temperature > 1 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            temperature=1.1
        )
    
    errors = exc_info.value.errors()
    assert any("temperature" in str(e.get("loc", [])) for e in errors)


def test_temperature_at_boundaries():
    """Verify temperature at 0.0 and 1.0 boundaries is valid."""
    # Temperature = 0.0 should be valid
    config = RuntimeConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        temperature=0.0
    )
    assert config.temperature == 0.0
    
    # Temperature = 1.0 should be valid
    config = RuntimeConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        temperature=1.0
    )
    assert config.temperature == 1.0


def test_max_tokens_negative():
    """Verify negative max_tokens raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            max_tokens=-1
        )
    
    errors = exc_info.value.errors()
    assert any("max_tokens" in str(e.get("loc", [])) for e in errors)


def test_max_tokens_zero():
    """Verify max_tokens=0 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            max_tokens=0
        )
    
    errors = exc_info.value.errors()
    assert any("max_tokens" in str(e.get("loc", [])) for e in errors)


def test_max_tokens_valid():
    """Verify positive max_tokens is accepted."""
    config = RuntimeConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096
    )
    assert config.max_tokens == 4096


def test_top_p_out_of_range_low():
    """Verify top_p < 0 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            top_p=-0.01
        )
    
    errors = exc_info.value.errors()
    assert any("top_p" in str(e.get("loc", [])) for e in errors)


def test_top_p_out_of_range_high():
    """Verify top_p > 1 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            top_p=1.01
        )
    
    errors = exc_info.value.errors()
    assert any("top_p" in str(e.get("loc", [])) for e in errors)


def test_top_p_at_boundaries():
    """Verify top_p at 0.0 and 1.0 boundaries is valid."""
    config = RuntimeConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        top_p=0.0
    )
    assert config.top_p == 0.0
    
    config = RuntimeConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        top_p=1.0
    )
    assert config.top_p == 1.0


def test_top_k_negative():
    """Verify negative top_k raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            top_k=-1
        )
    
    errors = exc_info.value.errors()
    assert any("top_k" in str(e.get("loc", [])) for e in errors)


def test_top_k_zero():
    """Verify top_k=0 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            top_k=0
        )
    
    errors = exc_info.value.errors()
    assert any("top_k" in str(e.get("loc", [])) for e in errors)


def test_top_k_valid():
    """Verify positive top_k is accepted."""
    config = RuntimeConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        top_k=40
    )
    assert config.top_k == 40


def test_stop_sequences_wrong_type():
    """Verify stop_sequences must be a list of strings."""
    with pytest.raises(ValidationError):
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            stop_sequences="not a list"  # type: ignore
        )


def test_stop_sequences_valid():
    """Verify stop_sequences list of strings is accepted."""
    config = RuntimeConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        stop_sequences=["STOP", "END"]
    )
    assert config.stop_sequences == ["STOP", "END"]


def test_metadata_wrong_type():
    """Verify metadata must be a dictionary."""
    with pytest.raises(ValidationError):
        RuntimeConfig(
            provider="claude",
            model="claude-3-5-sonnet-20241022",
            metadata="not a dict"  # type: ignore
        )


def test_metadata_valid():
    """Verify metadata dictionary is accepted."""
    config = RuntimeConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        metadata={"user_id": "test-user"}
    )
    assert config.metadata == {"user_id": "test-user"}


def test_all_claude_parameters_together():
    """Verify all Claude parameters can be set together."""
    config = RuntimeConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9,
        top_k=50,
        stop_sequences=["END"],
        metadata={"session": "test"}
    )
    
    assert config.temperature == 0.7
    assert config.max_tokens == 2048
    assert config.top_p == 0.9
    assert config.top_k == 50
    assert config.stop_sequences == ["END"]
    assert config.metadata == {"session": "test"}


def test_claude_parameters_default_to_none():
    """Verify Claude parameters default to None when not specified."""
    config = RuntimeConfig(
        provider="copilot",
        model="gpt-4"
    )
    
    assert config.temperature is None
    assert config.max_tokens is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.stop_sequences is None
    assert config.metadata is None
