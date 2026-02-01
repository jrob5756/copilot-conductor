"""Tests for Claude provider parse recovery mechanism.

Tests comprehensive failure scenarios for parse recovery:
- Malformed JSON in tool responses
- Missing required output fields
- Invalid JSON syntax
- Empty responses
- Nested JSON parsing errors
- Partial JSON fragments
- Multiple retry attempts
- Recovery success and failure paths
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from copilot_conductor.config.schema import AgentDef, OutputField
from copilot_conductor.exceptions import ExecutionError
from copilot_conductor.providers.claude import ClaudeProvider


class TestClaudeParseRecovery:
    """Tests for parse recovery with malformed responses."""

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_recovery_from_malformed_json_in_tool(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that parse recovery handles malformed JSON in tool responses."""
        mock_anthropic_module.__version__ = "0.77.0"
        
        # First response: malformed JSON
        malformed_message = Mock()
        malformed_message.id = "msg_malformed"
        malformed_message.content = [
            Mock(
                type="tool_use",
                id="tool_1",
                name="provide_output",
                input={"result": '{"answer": "incomplete...'},  # Malformed
            )
        ]
        malformed_message.stop_reason = "end_turn"
        malformed_message.usage = Mock(input_tokens=10, output_tokens=20)
        
        # Second response: corrected JSON
        corrected_message = Mock()
        corrected_message.id = "msg_corrected"
        corrected_message.content = [
            Mock(
                type="tool_use",
                id="tool_2",
                name="provide_output",
                input={"result": '{"answer": "Complete answer"}'},
            )
        ]
        corrected_message.stop_reason = "end_turn"
        corrected_message.usage = Mock(input_tokens=15, output_tokens=10)
        
        mock_client = Mock()
        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock(
            side_effect=[malformed_message, corrected_message]
        )
        mock_client.models = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        
        agent = AgentDef(
            name="test_agent",
            model="claude-3-5-sonnet-latest",
            prompt="Answer the question",
            output={"answer": OutputField(type="string")},
        )
        
        result = await provider.execute(agent, {"workflow": {"input": {}}})
        
        # Should successfully recover and return corrected output
        assert result["answer"] == "Complete answer"
        # Should have made 2 API calls (initial + recovery)
        assert mock_client.messages.create.call_count == 2

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_recovery_from_missing_required_fields(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test recovery when required output fields are missing."""
        mock_anthropic_module.__version__ = "0.77.0"
        
        # First response: missing required field
        incomplete_message = Mock()
        incomplete_message.id = "msg_incomplete"
        incomplete_message.content = [
            Mock(
                type="tool_use",
                id="tool_1",
                name="provide_output",
                input={"result": '{"wrong_field": "value"}'},
            )
        ]
        incomplete_message.stop_reason = "end_turn"
        incomplete_message.usage = Mock(input_tokens=10, output_tokens=15)
        
        # Second response: includes required field
        complete_message = Mock()
        complete_message.id = "msg_complete"
        complete_message.content = [
            Mock(
                type="tool_use",
                id="tool_2",
                name="provide_output",
                input={"result": '{"answer": "Correct answer", "summary": "Summary text"}'},
            )
        ]
        complete_message.stop_reason = "end_turn"
        complete_message.usage = Mock(input_tokens=12, output_tokens=18)
        
        mock_client = Mock()
        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock(
            side_effect=[incomplete_message, complete_message]
        )
        mock_client.models = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        
        agent = AgentDef(
            name="test_agent",
            model="claude-3-5-sonnet-latest",
            prompt="Provide structured answer",
            output={
                "answer": OutputField(type="string"),
                "summary": OutputField(type="string"),
            },
        )
        
        result = await provider.execute(agent, {"workflow": {"input": {}}})
        
        assert result["answer"] == "Correct answer"
        assert result["summary"] == "Summary text"
        assert mock_client.messages.create.call_count == 2

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_recovery_from_invalid_json_syntax(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test recovery from completely invalid JSON syntax."""
        mock_anthropic_module.__version__ = "0.77.0"
        
        # First response: invalid JSON
        invalid_message = Mock()
        invalid_message.id = "msg_invalid"
        invalid_message.content = [
            Mock(
                type="tool_use",
                id="tool_1",
                name="provide_output",
                input={"result": 'This is not JSON at all {{{'},
            )
        ]
        invalid_message.stop_reason = "end_turn"
        invalid_message.usage = Mock(input_tokens=10, output_tokens=10)
        
        # Second response: valid JSON
        valid_message = Mock()
        valid_message.id = "msg_valid"
        valid_message.content = [
            Mock(
                type="tool_use",
                id="tool_2",
                name="provide_output",
                input={"result": '{"answer": "Valid JSON response"}'},
            )
        ]
        valid_message.stop_reason = "end_turn"
        valid_message.usage = Mock(input_tokens=12, output_tokens=8)
        
        mock_client = Mock()
        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock(side_effect=[invalid_message, valid_message])
        mock_client.models = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        
        agent = AgentDef(
            name="test_agent",
            model="claude-3-5-sonnet-latest",
            prompt="Answer",
            output={"answer": OutputField(type="string")},
        )
        
        result = await provider.execute(agent, {"workflow": {"input": {}}})
        
        assert result["answer"] == "Valid JSON response"
        assert mock_client.messages.create.call_count == 2

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_recovery_failure_after_max_retries(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test that recovery fails gracefully after maximum retry attempts."""
        mock_anthropic_module.__version__ = "0.77.0"
        
        # All responses: malformed JSON
        malformed_message = Mock()
        malformed_message.id = "msg_malformed"
        malformed_message.content = [
            Mock(
                type="tool_use",
                id="tool_1",
                name="provide_output",
                input={"result": 'Invalid JSON'},
            )
        ]
        malformed_message.stop_reason = "end_turn"
        malformed_message.usage = Mock(input_tokens=10, output_tokens=5)
        
        mock_client = Mock()
        mock_client.messages = Mock()
        # Return malformed response repeatedly
        mock_client.messages.create = AsyncMock(return_value=malformed_message)
        mock_client.models = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        
        agent = AgentDef(
            name="test_agent",
            model="claude-3-5-sonnet-latest",
            prompt="Answer",
            output={"answer": OutputField(type="string")},
        )
        
        # Should raise ExecutionError after retries exhausted
        with pytest.raises(ExecutionError, match="Failed to extract structured output"):
            await provider.execute(agent, {"workflow": {"input": {}}})
        
        # Should have attempted multiple times (initial + retries)
        assert mock_client.messages.create.call_count > 1

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_recovery_from_empty_response(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test recovery when response is empty or whitespace."""
        mock_anthropic_module.__version__ = "0.77.0"
        
        # First response: empty
        empty_message = Mock()
        empty_message.id = "msg_empty"
        empty_message.content = [
            Mock(type="tool_use", id="tool_1", name="provide_output", input={"result": ""})
        ]
        empty_message.stop_reason = "end_turn"
        empty_message.usage = Mock(input_tokens=10, output_tokens=2)
        
        # Second response: valid
        valid_message = Mock()
        valid_message.id = "msg_valid"
        valid_message.content = [
            Mock(
                type="tool_use",
                id="tool_2",
                name="provide_output",
                input={"result": '{"answer": "Non-empty response"}'},
            )
        ]
        valid_message.stop_reason = "end_turn"
        valid_message.usage = Mock(input_tokens=12, output_tokens=10)
        
        mock_client = Mock()
        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock(side_effect=[empty_message, valid_message])
        mock_client.models = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        
        agent = AgentDef(
            name="test_agent",
            model="claude-3-5-sonnet-latest",
            prompt="Answer",
            output={"answer": OutputField(type="string")},
        )
        
        result = await provider.execute(agent, {"workflow": {"input": {}}})
        
        assert result["answer"] == "Non-empty response"
        assert mock_client.messages.create.call_count == 2

    @patch("copilot_conductor.providers.claude.ANTHROPIC_SDK_AVAILABLE", True)
    @patch("copilot_conductor.providers.claude.AsyncAnthropic")
    @patch("copilot_conductor.providers.claude.anthropic")
    @pytest.mark.asyncio
    async def test_fallback_to_text_content_parsing(
        self, mock_anthropic_module: Mock, mock_anthropic_class: Mock
    ) -> None:
        """Test fallback to parsing JSON from text content when tool_use fails."""
        mock_anthropic_module.__version__ = "0.77.0"
        
        # Response with JSON in text content (no tool_use)
        text_message = Mock()
        text_message.id = "msg_text"
        text_message.content = [
            Mock(
                type="text",
                text='Here is the answer: {"answer": "Parsed from text content"}',
            )
        ]
        text_message.stop_reason = "end_turn"
        text_message.usage = Mock(input_tokens=10, output_tokens=15)
        
        mock_client = Mock()
        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock(return_value=text_message)
        mock_client.models = Mock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_anthropic_class.return_value = mock_client

        provider = ClaudeProvider()
        
        agent = AgentDef(
            name="test_agent",
            model="claude-3-5-sonnet-latest",
            prompt="Answer",
            output={"answer": OutputField(type="string")},
        )
        
        result = await provider.execute(agent, {"workflow": {"input": {}}})
        
        # Should successfully extract JSON from text content
        assert result["answer"] == "Parsed from text content"
        # Should only make 1 call (fallback works on first try)
        assert mock_client.messages.create.call_count == 1
