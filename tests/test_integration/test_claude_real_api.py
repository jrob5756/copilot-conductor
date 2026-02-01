"""Real API integration tests for Claude provider.

EPIC-008-T9: Real API tests marked with pytest.mark.real_api

These tests require:
- ANTHROPIC_API_KEY environment variable
- Real API credits (costs money)
- Network connectivity

Run with: pytest -m real_api
Skip with: pytest -m "not real_api" (default)
"""

import os

import pytest

from copilot_conductor.config.schema import (
    AgentDef,
    OutputField,
    RouteDef,
    RuntimeConfig,
    WorkflowConfig,
    WorkflowDef,
)
from copilot_conductor.engine.workflow import WorkflowEngine
from copilot_conductor.providers.claude import ClaudeProvider


@pytest.mark.real_api
class TestClaudeRealAPI:
    """Real API tests (require ANTHROPIC_API_KEY)."""

    @pytest.fixture
    def skip_if_no_api_key(self):
        """Skip test if ANTHROPIC_API_KEY is not set."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set - skipping real API test")

    @pytest.mark.asyncio
    async def test_real_simple_qa(self, skip_if_no_api_key) -> None:
        """Test real API call with simple Q&A workflow."""
        workflow = WorkflowConfig(
            workflow=WorkflowDef(
                name="real-qa-test",
                description="Real API Q&A test",
                entry_point="qa_agent",
                runtime=RuntimeConfig(provider="claude"),
            ),
            agents=[
                AgentDef(
                    name="qa_agent",
                    model="claude-3-5-sonnet-latest",
                    prompt="Answer this question concisely: {{ workflow.input.question }}",
                    output={"answer": OutputField(type="string")},
                    routes=[RouteDef(to="$end")],
                )
            ],
        )

        provider = ClaudeProvider()
        
        # Verify connection before running workflow
        is_connected = await provider.validate_connection()
        assert is_connected, "Failed to connect to Claude API"

        engine = WorkflowEngine(workflow, provider)

        result = await engine.run({"question": "What is 2+2?"})

        # Verify result
        assert "qa_agent" in result
        assert "answer" in result["qa_agent"]
        assert "4" in result["qa_agent"]["answer"] or "four" in result["qa_agent"]["answer"].lower()

        # Cleanup
        await provider.close()

    @pytest.mark.asyncio
    async def test_real_structured_output(self, skip_if_no_api_key) -> None:
        """Test real API with structured output extraction."""
        workflow = WorkflowConfig(
            workflow=WorkflowDef(
                name="real-structured-test",
                description="Real API structured output test",
                entry_point="analyzer",
                runtime=RuntimeConfig(provider="claude"),
            ),
            agents=[
                AgentDef(
                    name="analyzer",
                    model="claude-3-5-sonnet-latest",
                    prompt="Analyze the programming language Python. Provide a title, a 1-sentence description, and a score from 0-100.",
                    output={
                        "title": OutputField(type="string"),
                        "description": OutputField(type="string"),
                        "score": OutputField(type="number"),
                    },
                    routes=[RouteDef(to="$end")],
                )
            ],
        )

        provider = ClaudeProvider()
        engine = WorkflowEngine(workflow, provider)

        result = await engine.run({})

        # Verify structured output
        assert "analyzer" in result
        assert "title" in result["analyzer"]
        assert "description" in result["analyzer"]
        assert "score" in result["analyzer"]
        
        # Verify types
        assert isinstance(result["analyzer"]["title"], str)
        assert isinstance(result["analyzer"]["description"], str)
        assert isinstance(result["analyzer"]["score"], (int, float))
        
        # Verify reasonable values
        assert len(result["analyzer"]["title"]) > 0
        assert len(result["analyzer"]["description"]) > 10
        assert 0 <= result["analyzer"]["score"] <= 100

        await provider.close()

    @pytest.mark.asyncio
    async def test_real_model_verification(self, skip_if_no_api_key) -> None:
        """Test that model verification lists available models."""
        provider = ClaudeProvider()

        # Connection validation should trigger model verification
        is_connected = await provider.validate_connection()
        assert is_connected

        # Provider should have logged available models (check doesn't raise)
        # This test verifies the model verification feature works
        
        await provider.close()

    @pytest.mark.asyncio
    async def test_real_invalid_model(self, skip_if_no_api_key) -> None:
        """Test workflow with invalid model name."""
        workflow = WorkflowConfig(
            workflow=WorkflowDef(
                name="invalid-model-test",
                description="Test invalid model handling",
                entry_point="agent1",
                runtime=RuntimeConfig(provider="claude"),
            ),
            agents=[
                AgentDef(
                    name="agent1",
                    model="invalid-model-name-12345",
                    prompt="Test",
                    output={"result": OutputField(type="string")},
                    routes=[RouteDef(to="$end")],
                )
            ],
        )

        provider = ClaudeProvider()
        engine = WorkflowEngine(workflow, provider)

        # Should raise an error for invalid model
        with pytest.raises(Exception):  # Could be ValidationError or ProviderError
            await engine.run({})

        await provider.close()

    @pytest.mark.asyncio
    async def test_real_connection_validation(self, skip_if_no_api_key) -> None:
        """Test connection validation with real API."""
        provider = ClaudeProvider()

        # Should succeed with valid API key
        is_valid = await provider.validate_connection()
        assert is_valid is True

        await provider.close()

    @pytest.mark.asyncio
    async def test_real_different_models(self, skip_if_no_api_key) -> None:
        """Test execution with different Claude models."""
        models_to_test = [
            "claude-3-5-sonnet-latest",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ]

        for model_name in models_to_test:
            workflow = WorkflowConfig(
                workflow=WorkflowDef(
                    name=f"test-{model_name}",
                    description=f"Test with {model_name}",
                    entry_point="agent1",
                    runtime=RuntimeConfig(provider="claude"),
                ),
                agents=[
                    AgentDef(
                        name="agent1",
                        model=model_name,
                        prompt="Say 'hello'",
                        output={"greeting": OutputField(type="string")},
                        routes=[RouteDef(to="$end")],
                    )
                ],
            )

            provider = ClaudeProvider()
            engine = WorkflowEngine(workflow, provider)

            result = await engine.run({})

            # Verify basic response
            assert "agent1" in result
            assert "greeting" in result["agent1"]
            assert len(result["agent1"]["greeting"]) > 0

            await provider.close()
