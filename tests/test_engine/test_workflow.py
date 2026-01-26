"""Integration tests for WorkflowEngine.

Tests cover:
- Linear workflow execution
- Context passing between agents
- Output template rendering
- Routing between agents
- Error handling
"""

import pytest

from copilot_conductor.config.schema import (
    AgentDef,
    ContextConfig,
    LimitsConfig,
    OutputField,
    RouteDef,
    RuntimeConfig,
    WorkflowConfig,
    WorkflowDef,
)
from copilot_conductor.engine.workflow import WorkflowEngine
from copilot_conductor.exceptions import ExecutionError
from copilot_conductor.providers.copilot import CopilotProvider


@pytest.fixture
def simple_workflow_config() -> WorkflowConfig:
    """Create a simple single-agent workflow config."""
    return WorkflowConfig(
        workflow=WorkflowDef(
            name="simple-workflow",
            entry_point="answerer",
            runtime=RuntimeConfig(provider="copilot"),
            context=ContextConfig(mode="accumulate"),
            limits=LimitsConfig(max_iterations=10),
        ),
        agents=[
            AgentDef(
                name="answerer",
                model="gpt-4",
                prompt="Answer: {{ workflow.input.question }}",
                output={"answer": OutputField(type="string")},
                routes=[RouteDef(to="$end")],
            ),
        ],
        output={
            "answer": "{{ answerer.output.answer }}",
        },
    )


@pytest.fixture
def multi_agent_workflow_config() -> WorkflowConfig:
    """Create a multi-agent workflow config."""
    return WorkflowConfig(
        workflow=WorkflowDef(
            name="multi-agent-workflow",
            entry_point="planner",
            runtime=RuntimeConfig(provider="copilot"),
            context=ContextConfig(mode="accumulate"),
            limits=LimitsConfig(max_iterations=10),
        ),
        agents=[
            AgentDef(
                name="planner",
                model="gpt-4",
                prompt="Plan for: {{ workflow.input.goal }}",
                output={"plan": OutputField(type="string")},
                routes=[RouteDef(to="executor")],
            ),
            AgentDef(
                name="executor",
                model="gpt-4",
                prompt="Execute plan: {{ planner.output.plan }}",
                output={"result": OutputField(type="string")},
                routes=[RouteDef(to="$end")],
            ),
        ],
        output={
            "plan": "{{ planner.output.plan }}",
            "result": "{{ executor.output.result }}",
        },
    )


@pytest.fixture
def conditional_workflow_config() -> WorkflowConfig:
    """Create a workflow with conditional routing."""
    return WorkflowConfig(
        workflow=WorkflowDef(
            name="conditional-workflow",
            entry_point="checker",
            runtime=RuntimeConfig(provider="copilot"),
            context=ContextConfig(mode="accumulate"),
            limits=LimitsConfig(max_iterations=10),
        ),
        agents=[
            AgentDef(
                name="checker",
                model="gpt-4",
                prompt="Check: {{ workflow.input.value }}",
                output={
                    "is_valid": OutputField(type="boolean"),
                    "message": OutputField(type="string"),
                },
                routes=[
                    RouteDef(to="success_handler", when="{{ output.is_valid }}"),
                    RouteDef(to="error_handler"),
                ],
            ),
            AgentDef(
                name="success_handler",
                model="gpt-4",
                prompt="Handle success: {{ checker.output.message }}",
                output={"result": OutputField(type="string")},
                routes=[RouteDef(to="$end")],
            ),
            AgentDef(
                name="error_handler",
                model="gpt-4",
                prompt="Handle error: {{ checker.output.message }}",
                output={"result": OutputField(type="string")},
                routes=[RouteDef(to="$end")],
            ),
        ],
        output={
            "result": "{{ context.history[-1] }}",
        },
    )


class TestWorkflowEngineBasic:
    """Basic WorkflowEngine tests."""

    @pytest.mark.asyncio
    async def test_simple_workflow_execution(
        self, simple_workflow_config: WorkflowConfig
    ) -> None:
        """Test executing a simple single-agent workflow."""

        def mock_handler(agent, prompt, context):
            return {"answer": "Python is a programming language"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(simple_workflow_config, provider)

        result = await engine.run({"question": "What is Python?"})

        assert "answer" in result
        assert result["answer"] == "Python is a programming language"

    @pytest.mark.asyncio
    async def test_multi_agent_workflow_execution(
        self, multi_agent_workflow_config: WorkflowConfig
    ) -> None:
        """Test executing a multi-agent workflow."""
        responses = {
            "planner": {"plan": "Step 1: Do X, Step 2: Do Y"},
            "executor": {"result": "Successfully completed X and Y"},
        }

        def mock_handler(agent, prompt, context):
            return responses[agent.name]

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(multi_agent_workflow_config, provider)

        result = await engine.run({"goal": "Complete the task"})

        assert result["plan"] == "Step 1: Do X, Step 2: Do Y"
        assert result["result"] == "Successfully completed X and Y"

    @pytest.mark.asyncio
    async def test_workflow_stores_context(
        self, multi_agent_workflow_config: WorkflowConfig
    ) -> None:
        """Test that workflow stores context between agents."""
        received_contexts = []

        def mock_handler(agent, prompt, context):
            received_contexts.append((agent.name, context.copy()))
            return {"plan": "the plan"} if agent.name == "planner" else {"result": "done"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(multi_agent_workflow_config, provider)

        await engine.run({"goal": "test"})

        # First agent should see workflow inputs
        assert received_contexts[0][0] == "planner"
        assert received_contexts[0][1]["workflow"]["input"]["goal"] == "test"

        # Second agent should see planner's output
        assert received_contexts[1][0] == "executor"
        assert received_contexts[1][1]["planner"]["output"]["plan"] == "the plan"


class TestWorkflowEngineContextModes:
    """Tests for different context accumulation modes."""

    @pytest.mark.asyncio
    async def test_last_only_mode(self) -> None:
        """Test last_only context mode."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="last-only-workflow",
                entry_point="agent1",
                context=ContextConfig(mode="last_only"),
            ),
            agents=[
                AgentDef(
                    name="agent1",
                    model="gpt-4",
                    prompt="First",
                    output={"out1": OutputField(type="string")},
                    routes=[RouteDef(to="agent2")],
                ),
                AgentDef(
                    name="agent2",
                    model="gpt-4",
                    prompt="Second",
                    output={"out2": OutputField(type="string")},
                    routes=[RouteDef(to="agent3")],
                ),
                AgentDef(
                    name="agent3",
                    model="gpt-4",
                    prompt="Third",
                    output={"out3": OutputField(type="string")},
                    routes=[RouteDef(to="$end")],
                ),
            ],
            output={"final": "{{ context.iteration }}"},
        )

        received_contexts = []

        def mock_handler(agent, prompt, context):
            received_contexts.append((agent.name, context.copy()))
            return {f"out{agent.name[-1]}": f"output_{agent.name[-1]}"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        await engine.run({})

        # Agent3 should only see agent2's output (last_only mode)
        agent3_context = received_contexts[2][1]
        assert "agent2" in agent3_context
        assert "agent1" not in agent3_context

    @pytest.mark.asyncio
    async def test_explicit_mode(self) -> None:
        """Test explicit context mode."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="explicit-workflow",
                entry_point="agent1",
                context=ContextConfig(mode="explicit"),
            ),
            agents=[
                AgentDef(
                    name="agent1",
                    model="gpt-4",
                    prompt="First",
                    input=["workflow.input.goal"],
                    output={"out1": OutputField(type="string")},
                    routes=[RouteDef(to="agent2")],
                ),
                AgentDef(
                    name="agent2",
                    model="gpt-4",
                    prompt="Second",
                    input=["agent1.output"],
                    output={"out2": OutputField(type="string")},
                    routes=[RouteDef(to="$end")],
                ),
            ],
            output={"result": "done"},
        )

        received_contexts = []

        def mock_handler(agent, prompt, context):
            received_contexts.append((agent.name, context.copy()))
            return {f"out{agent.name[-1]}": f"output_{agent.name[-1]}"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        await engine.run({"goal": "test", "other": "ignored"})

        # Agent2 should only see agent1's output (explicit mode)
        agent2_context = received_contexts[1][1]
        assert "agent1" in agent2_context
        # Workflow.input.goal should not be in agent2's context since it's not in input list
        assert "other" not in agent2_context.get("workflow", {}).get("input", {})


class TestWorkflowEngineRouting:
    """Tests for workflow routing."""

    @pytest.mark.asyncio
    async def test_conditional_route_true(
        self, conditional_workflow_config: WorkflowConfig
    ) -> None:
        """Test conditional routing when condition is true."""

        def mock_handler(agent, prompt, context):
            if agent.name == "checker":
                return {"is_valid": True, "message": "All good"}
            return {"result": f"Handled by {agent.name}"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(conditional_workflow_config, provider)

        await engine.run({"value": "test"})

        # Should have routed to success_handler
        assert "success_handler" in engine.context.execution_history
        assert "error_handler" not in engine.context.execution_history

    @pytest.mark.asyncio
    async def test_conditional_route_false(
        self, conditional_workflow_config: WorkflowConfig
    ) -> None:
        """Test conditional routing when condition is false."""

        def mock_handler(agent, prompt, context):
            if agent.name == "checker":
                return {"is_valid": False, "message": "Invalid"}
            return {"result": f"Handled by {agent.name}"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(conditional_workflow_config, provider)

        await engine.run({"value": "test"})

        # Should have routed to error_handler (fallthrough)
        assert "error_handler" in engine.context.execution_history
        assert "success_handler" not in engine.context.execution_history

    @pytest.mark.asyncio
    async def test_no_routes_ends_workflow(self) -> None:
        """Test that agent with no routes ends the workflow."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="no-routes",
                entry_point="agent1",
            ),
            agents=[
                AgentDef(
                    name="agent1",
                    model="gpt-4",
                    prompt="Hello",
                    output={"result": OutputField(type="string")},
                    routes=[],  # No routes
                ),
            ],
            output={"result": "{{ agent1.output.result }}"},
        )

        def mock_handler(agent, prompt, context):
            return {"result": "done"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        result = await engine.run({})

        assert result["result"] == "done"


class TestWorkflowEngineErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_missing_agent_raises_error(
        self, simple_workflow_config: WorkflowConfig
    ) -> None:
        """Test that missing entry point agent raises error."""
        simple_workflow_config.workflow.entry_point = "nonexistent"
        # Need to bypass the validation since we're modifying after creation
        simple_workflow_config.agents = simple_workflow_config.agents

        def mock_handler(agent, prompt, context):
            return {"answer": "test"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(simple_workflow_config, provider)

        with pytest.raises(ExecutionError, match="Agent not found"):
            await engine.run({"question": "test"})

    @pytest.mark.asyncio
    async def test_execution_summary(
        self, multi_agent_workflow_config: WorkflowConfig
    ) -> None:
        """Test getting execution summary."""

        def mock_handler(agent, prompt, context):
            if agent.name == "planner":
                return {"plan": "the plan"}
            return {"result": "done"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(multi_agent_workflow_config, provider)

        await engine.run({"goal": "test"})

        summary = engine.get_execution_summary()

        assert summary["iterations"] == 2
        assert summary["agents_executed"] == ["planner", "executor"]
        assert summary["context_mode"] == "accumulate"


class TestWorkflowEngineOutputTemplates:
    """Tests for output template rendering."""

    @pytest.mark.asyncio
    async def test_output_template_with_json(self) -> None:
        """Test output template with json filter."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="json-output",
                entry_point="agent1",
            ),
            agents=[
                AgentDef(
                    name="agent1",
                    model="gpt-4",
                    prompt="Hello",
                    output={
                        "data_list": OutputField(type="array"),
                    },
                    routes=[RouteDef(to="$end")],
                ),
            ],
            output={
                "data_list": "{{ agent1.output.data_list | json }}",
            },
        )

        def mock_handler(agent, prompt, context):
            return {"data_list": ["a", "b", "c"]}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        result = await engine.run({})

        # The json filter should produce valid JSON that gets parsed back
        assert result["data_list"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_output_template_numeric(self) -> None:
        """Test output template with numeric value."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="numeric-output",
                entry_point="agent1",
            ),
            agents=[
                AgentDef(
                    name="agent1",
                    model="gpt-4",
                    prompt="Count",
                    output={"count": OutputField(type="number")},
                    routes=[RouteDef(to="$end")],
                ),
            ],
            output={
                "total": "{{ agent1.output.count }}",
            },
        )

        def mock_handler(agent, prompt, context):
            return {"count": 42}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        result = await engine.run({})

        assert result["total"] == 42


class TestWorkflowEngineLoopBack:
    """Tests for loop-back routing patterns."""

    @pytest.mark.asyncio
    async def test_simple_loop_with_iteration_limit(self) -> None:
        """Test a simple loop that terminates after iterations."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="loop-workflow",
                entry_point="refiner",
                context=ContextConfig(mode="accumulate"),
                limits=LimitsConfig(max_iterations=10),
            ),
            agents=[
                AgentDef(
                    name="refiner",
                    model="gpt-4",
                    prompt="Refine iteration {{ context.iteration }}",
                    output={
                        "quality": OutputField(type="number"),
                        "result": OutputField(type="string"),
                    },
                    routes=[
                        RouteDef(to="$end", when="{{ output.quality >= 8 }}"),
                        RouteDef(to="refiner"),  # Loop back
                    ],
                ),
            ],
            output={
                "result": "{{ refiner.output.result }}",
                "iterations": "{{ context.iteration }}",
            },
        )

        iteration_count = 0

        def mock_handler(agent, prompt, context):
            nonlocal iteration_count
            iteration_count += 1
            # Quality improves with each iteration
            quality = 5 + iteration_count
            return {"quality": quality, "result": f"Refined {iteration_count} times"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        result = await engine.run({})

        # Should have looped until quality >= 8
        assert iteration_count == 3  # quality: 6, 7, 8
        assert result["result"] == "Refined 3 times"

    @pytest.mark.asyncio
    async def test_loop_with_arithmetic_condition(self) -> None:
        """Test loop with arithmetic expression condition."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="arithmetic-loop",
                entry_point="counter",
            ),
            agents=[
                AgentDef(
                    name="counter",
                    model="gpt-4",
                    prompt="Count",
                    output={"count": OutputField(type="number")},
                    routes=[
                        RouteDef(to="$end", when="count >= 3"),
                        RouteDef(to="counter"),
                    ],
                ),
            ],
            output={"final_count": "{{ counter.output.count }}"},
        )

        call_count = 0

        def mock_handler(agent, prompt, context):
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        result = await engine.run({})

        assert call_count == 3
        assert result["final_count"] == 3

    @pytest.mark.asyncio
    async def test_loop_back_to_different_agent(self) -> None:
        """Test loop-back to a different previously executed agent."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="multi-agent-loop",
                entry_point="drafter",
            ),
            agents=[
                AgentDef(
                    name="drafter",
                    model="gpt-4",
                    prompt="Draft content",
                    output={"draft": OutputField(type="string")},
                    routes=[RouteDef(to="reviewer")],
                ),
                AgentDef(
                    name="reviewer",
                    model="gpt-4",
                    prompt="Review: {{ drafter.output.draft }}",
                    output={
                        "approved": OutputField(type="boolean"),
                        "feedback": OutputField(type="string"),
                    },
                    routes=[
                        RouteDef(to="$end", when="{{ output.approved }}"),
                        RouteDef(to="drafter"),  # Loop back to drafter
                    ],
                ),
            ],
            output={"final_draft": "{{ drafter.output.draft }}"},
        )

        loop_count = 0

        def mock_handler(agent, prompt, context):
            nonlocal loop_count
            if agent.name == "drafter":
                loop_count += 1
                return {"draft": f"Draft v{loop_count}"}
            else:  # reviewer
                # Approve on second review
                return {
                    "approved": loop_count >= 2,
                    "feedback": "Needs work" if loop_count < 2 else "Approved",
                }

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        result = await engine.run({})

        assert loop_count == 2
        assert result["final_draft"] == "Draft v2"
        # Verify execution history shows the loop
        summary = engine.get_execution_summary()
        assert summary["agents_executed"] == ["drafter", "reviewer", "drafter", "reviewer"]

    @pytest.mark.asyncio
    async def test_iteration_tracking_across_loops(self) -> None:
        """Test that iteration count is tracked across loop iterations."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="iteration-tracking",
                entry_point="agent",
            ),
            agents=[
                AgentDef(
                    name="agent",
                    model="gpt-4",
                    prompt="Iteration: {{ context.iteration }}",
                    output={"done": OutputField(type="boolean")},
                    routes=[
                        RouteDef(to="$end", when="{{ output.done }}"),
                        RouteDef(to="agent"),
                    ],
                ),
            ],
            output={"total_iterations": "{{ context.iteration }}"},
        )

        received_iterations = []

        def mock_handler(agent, prompt, context):
            received_iterations.append(context.get("context", {}).get("iteration", 0))
            return {"done": len(received_iterations) >= 3}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        result = await engine.run({})

        # Iterations should be tracked
        assert len(received_iterations) == 3
        assert result["total_iterations"] == 3


class TestWorkflowEngineRouterIntegration:
    """Tests for Router integration with WorkflowEngine."""

    @pytest.mark.asyncio
    async def test_route_output_transform(self) -> None:
        """Test that route output transforms are applied on $end."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="output-transform",
                entry_point="processor",
            ),
            agents=[
                AgentDef(
                    name="processor",
                    model="gpt-4",
                    prompt="Process",
                    output={"value": OutputField(type="string")},
                    routes=[
                        RouteDef(
                            to="$end",
                            output={"transformed": "Transformed: {{ output.value }}"},
                        ),
                    ],
                ),
            ],
            output={
                "original": "{{ processor.output.value }}",
            },
        )

        def mock_handler(agent, prompt, context):
            return {"value": "test_value"}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        result = await engine.run({})

        assert result["original"] == "test_value"
        assert result["transformed"] == "Transformed: test_value"

    @pytest.mark.asyncio
    async def test_no_matching_routes_error(self) -> None:
        """Test that no matching routes raises clear error."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="no-match",
                entry_point="agent",
            ),
            agents=[
                AgentDef(
                    name="agent",
                    model="gpt-4",
                    prompt="Test",
                    output={"flag": OutputField(type="boolean")},
                    routes=[
                        RouteDef(to="$end", when="{{ output.flag }}"),
                        # Missing catch-all route!
                    ],
                ),
            ],
            output={},
        )

        def mock_handler(agent, prompt, context):
            return {"flag": False}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        with pytest.raises(ValueError, match="No matching route found"):
            await engine.run({})

    @pytest.mark.asyncio
    async def test_mixed_jinja_and_arithmetic_conditions(self) -> None:
        """Test workflow with mixed Jinja2 and arithmetic conditions."""
        config = WorkflowConfig(
            workflow=WorkflowDef(
                name="mixed-conditions",
                entry_point="evaluator",
            ),
            agents=[
                AgentDef(
                    name="evaluator",
                    model="gpt-4",
                    prompt="Evaluate",
                    output={
                        "score": OutputField(type="number"),
                        "valid": OutputField(type="boolean"),
                    },
                    routes=[
                        RouteDef(to="high", when="score >= 8"),  # arithmetic
                        RouteDef(to="valid", when="{{ output.valid }}"),  # jinja
                        RouteDef(to="default"),
                    ],
                ),
                AgentDef(
                    name="high",
                    model="gpt-4",
                    prompt="High score",
                    routes=[RouteDef(to="$end")],
                    output={"result": OutputField(type="string")},
                ),
                AgentDef(
                    name="valid",
                    model="gpt-4",
                    prompt="Valid",
                    routes=[RouteDef(to="$end")],
                    output={"result": OutputField(type="string")},
                ),
                AgentDef(
                    name="default",
                    model="gpt-4",
                    prompt="Default",
                    routes=[RouteDef(to="$end")],
                    output={"result": OutputField(type="string")},
                ),
            ],
            output={"path": "{{ context.history[-1] }}"},
        )

        def mock_handler(agent, prompt, context):
            if agent.name == "evaluator":
                return {"score": 5, "valid": True}
            return {"result": agent.name}

        provider = CopilotProvider(mock_handler=mock_handler)
        engine = WorkflowEngine(config, provider)

        await engine.run({})

        # Score 5 < 8, but valid is True, so should go to 'valid'
        assert "valid" in engine.context.execution_history
