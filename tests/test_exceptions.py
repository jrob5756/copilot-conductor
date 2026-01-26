"""Test that the exceptions module works correctly."""

from copilot_conductor.exceptions import (
    ConductorError,
    ConfigurationError,
    ExecutionError,
    HumanGateError,
    MaxIterationsError,
    ProviderError,
    TemplateError,
    TimeoutError,
    ValidationError,
)


class TestConductorError:
    """Tests for the base ConductorError class."""

    def test_basic_error_message(self) -> None:
        """Test that basic error message is preserved."""
        error = ConductorError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_error_with_suggestion(self) -> None:
        """Test that error message includes suggestion when provided."""
        error = ConductorError("Something went wrong", suggestion="Try doing X instead")
        assert "Something went wrong" in str(error)
        assert "💡 Suggestion: Try doing X instead" in str(error)

    def test_suggestion_attribute(self) -> None:
        """Test that suggestion attribute is accessible."""
        error = ConductorError("Error", suggestion="Fix it")
        assert error.suggestion == "Fix it"

    def test_no_suggestion(self) -> None:
        """Test that suggestion is None when not provided."""
        error = ConductorError("Error")
        assert error.suggestion is None


class TestSpecificErrors:
    """Tests for specific error types."""

    def test_configuration_error(self) -> None:
        """Test ConfigurationError inherits from ConductorError."""
        error = ConfigurationError("Bad config")
        assert isinstance(error, ConductorError)
        assert str(error) == "Bad config"

    def test_validation_error(self) -> None:
        """Test ValidationError inherits from ConductorError."""
        error = ValidationError("Invalid data")
        assert isinstance(error, ConductorError)

    def test_template_error(self) -> None:
        """Test TemplateError inherits from ConductorError."""
        error = TemplateError("Template syntax error")
        assert isinstance(error, ConductorError)

    def test_provider_error(self) -> None:
        """Test ProviderError inherits from ConductorError."""
        error = ProviderError("Provider failed")
        assert isinstance(error, ConductorError)

    def test_execution_error(self) -> None:
        """Test ExecutionError inherits from ConductorError."""
        error = ExecutionError("Execution failed")
        assert isinstance(error, ConductorError)


class TestMaxIterationsError:
    """Tests for MaxIterationsError."""

    def test_inherits_from_execution_error(self) -> None:
        """Test MaxIterationsError inherits from ExecutionError."""
        error = MaxIterationsError(
            "Too many iterations",
            iterations=10,
            max_iterations=10,
        )
        assert isinstance(error, ExecutionError)
        assert isinstance(error, ConductorError)

    def test_attributes_are_set(self) -> None:
        """Test that all attributes are properly set."""
        error = MaxIterationsError(
            "Too many iterations",
            iterations=5,
            max_iterations=10,
            agent_history=["agent1", "agent2", "agent1"],
            suggestion="Increase max_iterations",
        )
        assert error.iterations == 5
        assert error.max_iterations == 10
        assert error.agent_history == ["agent1", "agent2", "agent1"]
        assert error.suggestion == "Increase max_iterations"

    def test_default_agent_history(self) -> None:
        """Test that agent_history defaults to empty list."""
        error = MaxIterationsError(
            "Too many iterations",
            iterations=10,
            max_iterations=10,
        )
        assert error.agent_history == []


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_inherits_from_execution_error(self) -> None:
        """Test TimeoutError inherits from ExecutionError."""
        error = TimeoutError(
            "Workflow timed out",
            elapsed_seconds=600.0,
            timeout_seconds=600.0,
        )
        assert isinstance(error, ExecutionError)
        assert isinstance(error, ConductorError)

    def test_attributes_are_set(self) -> None:
        """Test that all attributes are properly set."""
        error = TimeoutError(
            "Workflow timed out",
            elapsed_seconds=300.5,
            timeout_seconds=600.0,
            current_agent="reviewer",
            suggestion="Increase timeout_seconds",
        )
        assert error.elapsed_seconds == 300.5
        assert error.timeout_seconds == 600.0
        assert error.current_agent == "reviewer"
        assert error.suggestion == "Increase timeout_seconds"

    def test_default_current_agent(self) -> None:
        """Test that current_agent defaults to None."""
        error = TimeoutError(
            "Workflow timed out",
            elapsed_seconds=600.0,
            timeout_seconds=600.0,
        )
        assert error.current_agent is None


class TestHumanGateError:
    """Tests for HumanGateError."""

    def test_inherits_from_execution_error(self) -> None:
        """Test HumanGateError inherits from ExecutionError."""
        error = HumanGateError("User cancelled")
        assert isinstance(error, ExecutionError)
        assert isinstance(error, ConductorError)

    def test_with_suggestion(self) -> None:
        """Test HumanGateError with suggestion."""
        error = HumanGateError("Invalid option", suggestion="Choose a valid option")
        assert "Invalid option" in str(error)
        assert "💡 Suggestion: Choose a valid option" in str(error)
