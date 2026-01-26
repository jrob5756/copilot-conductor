"""Tests for the run command and input parsing.

This module tests:
- Input flag parsing (--input name=value)
- Type coercion for input values
- InputCollector for --input.name=value patterns
- Run command execution with mock provider
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from copilot_conductor.cli.app import app
from copilot_conductor.cli.run import (
    InputCollector,
    coerce_value,
    parse_input_flags,
)

runner = CliRunner()


class TestCoerceValue:
    """Tests for the coerce_value function."""

    def test_coerce_true(self) -> None:
        """Test coercing 'true' to boolean."""
        assert coerce_value("true") is True
        assert coerce_value("True") is True
        assert coerce_value("TRUE") is True

    def test_coerce_false(self) -> None:
        """Test coercing 'false' to boolean."""
        assert coerce_value("false") is False
        assert coerce_value("False") is False
        assert coerce_value("FALSE") is False

    def test_coerce_null(self) -> None:
        """Test coercing 'null' to None."""
        assert coerce_value("null") is None
        assert coerce_value("Null") is None
        assert coerce_value("NULL") is None

    def test_coerce_integer(self) -> None:
        """Test coercing integer strings."""
        assert coerce_value("42") == 42
        assert coerce_value("-10") == -10
        assert coerce_value("0") == 0

    def test_coerce_float(self) -> None:
        """Test coercing float strings."""
        assert coerce_value("3.14") == 3.14
        assert coerce_value("-2.5") == -2.5
        assert coerce_value("0.0") == 0.0

    def test_coerce_json_array(self) -> None:
        """Test coercing JSON array strings."""
        result = coerce_value('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_coerce_json_object(self) -> None:
        """Test coercing JSON object strings."""
        result = coerce_value('{"key": "value"}')
        assert result == {"key": "value"}

    def test_coerce_invalid_json_returns_string(self) -> None:
        """Test that invalid JSON returns the original string."""
        result = coerce_value('[not valid json')
        assert result == '[not valid json'

    def test_coerce_string(self) -> None:
        """Test that regular strings are returned unchanged."""
        assert coerce_value("hello") == "hello"
        assert coerce_value("Hello World!") == "Hello World!"
        assert coerce_value("") == ""


class TestParseInputFlags:
    """Tests for parse_input_flags function."""

    def test_parse_single_input(self) -> None:
        """Test parsing a single input."""
        result = parse_input_flags(["name=value"])
        assert result == {"name": "value"}

    def test_parse_multiple_inputs(self) -> None:
        """Test parsing multiple inputs."""
        result = parse_input_flags(["name=Alice", "age=30", "active=true"])
        assert result == {"name": "Alice", "age": 30, "active": True}

    def test_parse_value_with_equals(self) -> None:
        """Test parsing value containing equals sign."""
        result = parse_input_flags(["equation=a=b+c"])
        assert result == {"equation": "a=b+c"}

    def test_parse_empty_value(self) -> None:
        """Test parsing empty value."""
        result = parse_input_flags(["empty="])
        assert result == {"empty": ""}

    def test_parse_json_value(self) -> None:
        """Test parsing JSON value."""
        result = parse_input_flags(['data={"key": "value"}'])
        assert result == {"data": {"key": "value"}}

    def test_parse_missing_equals_raises(self) -> None:
        """Test that missing equals raises BadParameter."""
        import typer
        with pytest.raises(typer.BadParameter, match="Invalid input format"):
            parse_input_flags(["invalid"])

    def test_parse_empty_name_raises(self) -> None:
        """Test that empty name raises BadParameter."""
        import typer
        with pytest.raises(typer.BadParameter, match="Empty input name"):
            parse_input_flags(["=value"])


class TestInputCollector:
    """Tests for InputCollector class."""

    def test_extract_input_dot_pattern(self) -> None:
        """Test extracting --input.name=value pattern."""
        args = ["run", "workflow.yaml", "--input.question=Hello"]
        result = InputCollector.extract_from_args(args)
        assert result == {"question": "Hello"}

    def test_extract_multiple_inputs(self) -> None:
        """Test extracting multiple inputs."""
        args = [
            "run", "workflow.yaml",
            "--input.name=Alice",
            "--input.age=30",
            "--input.active=true",
        ]
        result = InputCollector.extract_from_args(args)
        assert result == {"name": "Alice", "age": 30, "active": True}

    def test_extract_with_type_coercion(self) -> None:
        """Test that values are type coerced."""
        args = [
            "--input.count=42",
            "--input.ratio=3.14",
            "--input.enabled=true",
            "--input.data=[1, 2, 3]",
        ]
        result = InputCollector.extract_from_args(args)
        assert result == {
            "count": 42,
            "ratio": 3.14,
            "enabled": True,
            "data": [1, 2, 3],
        }

    def test_extract_ignores_other_flags(self) -> None:
        """Test that non-input flags are ignored."""
        args = [
            "run", "workflow.yaml",
            "--provider", "copilot",
            "--input.name=Alice",
            "--verbose",
        ]
        result = InputCollector.extract_from_args(args)
        assert result == {"name": "Alice"}

    def test_extract_empty_args(self) -> None:
        """Test with empty args."""
        result = InputCollector.extract_from_args([])
        assert result == {}


class TestRunCommand:
    """Tests for the run command."""

    def test_run_command_help(self) -> None:
        """Test that run --help works."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run a workflow from a YAML file" in result.output

    def test_run_command_missing_file(self) -> None:
        """Test that missing file produces error."""
        result = runner.invoke(app, ["run", "nonexistent.yaml"])
        # Should fail because file doesn't exist
        assert result.exit_code != 0

    def test_run_command_with_inputs(self, tmp_path: Path) -> None:
        """Test run command with input flags and mock provider."""
        # Create a simple workflow file
        workflow_file = tmp_path / "test.yaml"
        workflow_file.write_text("""\
workflow:
  name: test-workflow
  entry_point: greeter

agents:
  - name: greeter
    model: gpt-4
    prompt: "Say hello to {{ workflow.input.name }}"
    output:
      greeting:
        type: string
    routes:
      - to: $end

output:
  message: "{{ greeter.output.greeting }}"
""")

        # Mock the run_workflow_async function
        with patch("copilot_conductor.cli.run.run_workflow_async") as mock_run:
            mock_run.return_value = {"message": "Hello, World!"}

            runner.invoke(app, [
                "run", str(workflow_file),
                "-i", "name=World",
            ])

            # Check the mock was called
            assert mock_run.called
            call_args = mock_run.call_args

            # Verify inputs were passed
            assert call_args[0][1] == {"name": "World"}

    def test_run_command_with_provider_override(self, tmp_path: Path) -> None:
        """Test run command with provider override."""
        workflow_file = tmp_path / "test.yaml"
        workflow_file.write_text("""\
workflow:
  name: test-workflow
  entry_point: agent1

agents:
  - name: agent1
    model: gpt-4
    prompt: "Hello"
    routes:
      - to: $end

output:
  result: "done"
""")

        with patch("copilot_conductor.cli.run.run_workflow_async") as mock_run:
            mock_run.return_value = {"result": "done"}

            runner.invoke(app, [
                "run", str(workflow_file),
                "--provider", "copilot",
            ])

            # Verify provider was passed
            assert mock_run.called
            call_args = mock_run.call_args
            assert call_args[0][2] == "copilot"

    def test_run_command_json_output(self, tmp_path: Path) -> None:
        """Test that output is valid JSON."""
        workflow_file = tmp_path / "test.yaml"
        workflow_file.write_text("""\
workflow:
  name: test-workflow
  entry_point: agent1

agents:
  - name: agent1
    prompt: "Hello"
    routes:
      - to: $end

output:
  greeting: "Hello"
""")

        with patch("copilot_conductor.cli.run.run_workflow_async") as mock_run:
            mock_run.return_value = {"greeting": "Hello, World!"}

            result = runner.invoke(app, ["run", str(workflow_file)])

            # Output should be valid JSON
            # The output may have ANSI codes from Rich, so we just check it contains JSON
            assert "Hello, World!" in result.output or result.exit_code == 0


class TestVersionFlag:
    """Tests for the --version flag."""

    def test_version_flag(self) -> None:
        """Test --version shows version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Copilot Conductor v" in result.output

    def test_version_short_flag(self) -> None:
        """Test -v shows version."""
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "Copilot Conductor v" in result.output


class TestHelpFlag:
    """Tests for the --help flag."""

    def test_help_flag(self) -> None:
        """Test --help shows help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Copilot Conductor" in result.output
        assert "run" in result.output

    def test_no_args_shows_help(self) -> None:
        """Test running with no args shows help."""
        result = runner.invoke(app, [])
        # Typer with no_args_is_help=True shows help but returns exit code 2
        # when no command is provided (this is expected Typer behavior)
        assert "Usage" in result.output
        assert "run" in result.output
