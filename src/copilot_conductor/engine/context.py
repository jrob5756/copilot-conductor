"""Workflow context management for Copilot Conductor.

This module provides the WorkflowContext class for managing workflow state,
including inputs, agent outputs, and execution history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from copilot_conductor.providers.base import AgentProvider

# Token estimation constants
# Average characters per token (conservative estimate)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple character-based estimate. For more accurate token
    counting, use tiktoken or the provider's tokenizer.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    return len(text) // CHARS_PER_TOKEN


def estimate_dict_tokens(data: dict[str, Any]) -> int:
    """Estimate the number of tokens in a dictionary (serialized as JSON).

    Args:
        data: The dictionary to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    import json

    try:
        text = json.dumps(data)
        return estimate_tokens(text)
    except (TypeError, ValueError):
        # If we can't serialize, estimate from string representation
        return estimate_tokens(str(data))


@dataclass
class WorkflowContext:
    """Manages workflow execution state and context accumulation.

    The WorkflowContext stores all state needed during workflow execution:
    - workflow_inputs: Initial inputs provided when starting the workflow
    - agent_outputs: Outputs from each executed agent
    - current_iteration: Counter for iteration limit enforcement
    - execution_history: Ordered list of executed agent names

    Context modes:
    - accumulate: All prior agent outputs are available to subsequent agents
    - last_only: Only the most recent agent's output is available
    - explicit: Only inputs explicitly declared in the agent's input list are available

    Example:
        >>> ctx = WorkflowContext()
        >>> ctx.set_workflow_inputs({"question": "What is Python?"})
        >>> ctx.store("answerer", {"answer": "A programming language"})
        >>> agent_ctx = ctx.build_for_agent("checker", ["answerer.output"], "accumulate")
        >>> agent_ctx["answerer"]["output"]["answer"]
        'A programming language'
    """

    workflow_inputs: dict[str, Any] = field(default_factory=dict)
    """Inputs provided at workflow start."""

    agent_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Outputs from executed agents, keyed by agent name."""

    current_iteration: int = 0
    """Current execution iteration count."""

    execution_history: list[str] = field(default_factory=list)
    """Ordered list of executed agent names."""

    def set_workflow_inputs(self, inputs: dict[str, Any]) -> None:
        """Store workflow-level inputs.

        Args:
            inputs: Dictionary of input values provided at workflow start.
        """
        self.workflow_inputs = inputs.copy()

    def store(self, agent_name: str, output: dict[str, Any]) -> None:
        """Store an agent's output in context.

        This method updates the agent_outputs dictionary, appends the agent
        to execution_history, and increments the iteration counter.

        Args:
            agent_name: The name of the agent whose output is being stored.
            output: The structured output from the agent.
        """
        self.agent_outputs[agent_name] = output
        self.execution_history.append(agent_name)
        self.current_iteration += 1

    def build_for_agent(
        self,
        agent_name: str,
        inputs: list[str],
        mode: str = "accumulate",
    ) -> dict[str, Any]:
        """Build context dict for a specific agent based on its input declarations.

        The context includes:
        - workflow: Contains workflow-level inputs under workflow.input
        - context: Metadata about execution (iteration, history)
        - Agent outputs: Based on the accumulation mode

        Args:
            agent_name: Name of the agent needing context.
            inputs: List of input references (e.g., ['workflow.input.goal', 'planner.output']).
            mode: Context mode - accumulate, last_only, or explicit.

        Returns:
            Dict with 'workflow', agent outputs, and 'context' metadata.

        Raises:
            KeyError: If explicit mode is used and a required (non-optional) input is missing.
        """
        # For explicit mode, start with empty workflow inputs
        # For other modes, include all workflow inputs
        if mode == "explicit":
            ctx: dict[str, Any] = {
                "workflow": {"input": {}},
                "context": {
                    "iteration": self.current_iteration,
                    "history": self.execution_history.copy(),
                },
            }
            # Only declared inputs - parse input references
            for input_ref in inputs:
                self._add_explicit_input(ctx, input_ref)
        else:
            ctx = {
                "workflow": {"input": self.workflow_inputs.copy()},
                "context": {
                    "iteration": self.current_iteration,
                    "history": self.execution_history.copy(),
                },
            }

            if mode == "accumulate":
                # All prior agent outputs available
                for agent, output in self.agent_outputs.items():
                    ctx[agent] = {"output": output}

            elif mode == "last_only" and self.execution_history:
                # Only the most recent agent's output
                last_agent = self.execution_history[-1]
                ctx[last_agent] = {"output": self.agent_outputs.get(last_agent, {})}

        return ctx

    def _add_explicit_input(self, ctx: dict[str, Any], input_ref: str) -> None:
        """Add an explicit input reference to the context.

        Handles optional dependencies with '?' suffix - missing optional
        dependencies are silently skipped instead of raising errors.

        Input reference formats:
        - workflow.input.param_name - References a workflow input
        - agent_name.output - References an agent's entire output
        - agent_name.output.field - References a specific output field
        - agent_name.field - Shorthand for agent_name.output.field (deprecated but supported)
        - Any reference with '?' suffix - Optional dependency

        Args:
            ctx: The context dictionary to update.
            input_ref: The input reference string.

        Raises:
            KeyError: If a required (non-optional) input is not found.
        """
        # Check for optional suffix
        is_optional = input_ref.endswith("?")
        ref = input_ref.rstrip("?")

        parts = ref.split(".")

        if len(parts) < 2:
            if not is_optional:
                raise KeyError(f"Invalid input reference format: {input_ref}")
            return

        if parts[0] == "workflow":
            # workflow.input.param_name format
            if len(parts) >= 3 and parts[1] == "input":
                param_name = parts[2]
                # Ensure workflow.input exists in ctx
                if "workflow" not in ctx:
                    ctx["workflow"] = {"input": {}}
                elif "input" not in ctx["workflow"]:
                    ctx["workflow"]["input"] = {}

                if param_name in self.workflow_inputs:
                    ctx["workflow"]["input"][param_name] = self.workflow_inputs[param_name]
                elif is_optional:
                    # Set optional inputs to None so templates can check them
                    ctx["workflow"]["input"][param_name] = None
                else:
                    raise KeyError(f"Missing required workflow input: {param_name}")
        else:
            # agent_name.output or agent_name.output.field or agent_name.field format
            agent_name = parts[0]

            if agent_name in self.agent_outputs:
                # Ensure the agent context exists
                if agent_name not in ctx:
                    ctx[agent_name] = {"output": {}}
                elif "output" not in ctx[agent_name]:
                    ctx[agent_name]["output"] = {}

                # If just agent_name.output, copy entire output
                if len(parts) == 2 and parts[1] == "output":
                    ctx[agent_name]["output"] = self.agent_outputs[agent_name].copy()
                elif len(parts) >= 3 and parts[1] == "output":
                    # Copy specific field(s): agent_name.output.field
                    field_name = parts[2]
                    if field_name in self.agent_outputs[agent_name]:
                        ctx[agent_name]["output"][field_name] = self.agent_outputs[
                            agent_name
                        ][field_name]
                    elif not is_optional:
                        raise KeyError(
                            f"Missing output field '{field_name}' from agent '{agent_name}'"
                        )
                elif len(parts) == 2 and parts[1] != "output":
                    # Shorthand format: agent_name.field -> agent_name.output.field
                    field_name = parts[1]
                    if field_name in self.agent_outputs[agent_name]:
                        ctx[agent_name]["output"][field_name] = self.agent_outputs[
                            agent_name
                        ][field_name]
                    elif not is_optional:
                        raise KeyError(
                            f"Missing output field '{field_name}' from agent '{agent_name}'"
                        )
            elif not is_optional:
                raise KeyError(f"Missing required agent output: {agent_name}")

    def get_for_template(self) -> dict[str, Any]:
        """Get full context for template rendering.

        Returns a context dictionary with all agent outputs and workflow
        inputs available for use in output template expressions.

        Returns:
            Dict with workflow inputs and all agent outputs.
        """
        return self.build_for_agent("__template__", [], mode="accumulate")

    def get_latest_output(self) -> dict[str, Any] | None:
        """Get the output from the most recently executed agent.

        Returns:
            The output dictionary from the last agent, or None if no agents executed.
        """
        if not self.execution_history:
            return None
        last_agent = self.execution_history[-1]
        return self.agent_outputs.get(last_agent)

    def estimate_context_tokens(self) -> int:
        """Estimate the total number of tokens in the current context.

        Returns:
            Estimated number of tokens in the full context.
        """
        ctx = self.get_for_template()
        return estimate_dict_tokens(ctx)

    def trim_context(
        self,
        max_tokens: int,
        strategy: Literal["truncate", "drop_oldest", "summarize"] = "drop_oldest",
        provider: AgentProvider | None = None,
    ) -> int:
        """Trim context to fit within max_tokens.

        Applies the specified trimming strategy to reduce context size.

        Strategies:
        - truncate: Cut oldest content from each output to fit
        - drop_oldest: Remove entire oldest agent outputs FIFO until within limit
        - summarize: Use LLM provider to summarize context (requires provider)

        Args:
            max_tokens: Maximum number of tokens allowed.
            strategy: Trimming strategy to use.
            provider: Provider for summarize strategy (required for summarize).

        Returns:
            Number of tokens after trimming.

        Raises:
            ValueError: If summarize strategy is used without a provider.
        """
        current_tokens = self.estimate_context_tokens()

        if current_tokens <= max_tokens:
            return current_tokens

        if strategy == "drop_oldest":
            return self._trim_drop_oldest(max_tokens)
        elif strategy == "truncate":
            return self._trim_truncate(max_tokens)
        elif strategy == "summarize":
            if provider is None:
                raise ValueError("summarize strategy requires a provider")
            return self._trim_summarize(max_tokens, provider)
        else:
            raise ValueError(f"Unknown trimming strategy: {strategy}")

    def _trim_drop_oldest(self, max_tokens: int) -> int:
        """Trim by dropping oldest agent outputs.

        Removes entire agent outputs from oldest to newest until
        the context fits within max_tokens.

        Args:
            max_tokens: Maximum number of tokens allowed.

        Returns:
            Number of tokens after trimming.
        """
        # Get unique agents in execution order (first occurrence)
        seen_agents: list[str] = []
        for agent in self.execution_history:
            if agent not in seen_agents:
                seen_agents.append(agent)

        # Drop from oldest to newest
        for agent_name in seen_agents:
            if self.estimate_context_tokens() <= max_tokens:
                break

            if agent_name in self.agent_outputs:
                del self.agent_outputs[agent_name]

        return self.estimate_context_tokens()

    def _trim_truncate(self, max_tokens: int) -> int:
        """Trim by truncating oldest content from outputs.

        Truncates string values in agent outputs, starting from the
        oldest agent, until the context fits within max_tokens.

        Args:
            max_tokens: Maximum number of tokens allowed.

        Returns:
            Number of tokens after trimming.
        """
        # Get unique agents in execution order
        seen_agents: list[str] = []
        for agent in self.execution_history:
            if agent not in seen_agents:
                seen_agents.append(agent)

        # Calculate how many tokens we need to cut
        current_tokens = self.estimate_context_tokens()
        tokens_to_cut = current_tokens - max_tokens

        if tokens_to_cut <= 0:
            return current_tokens

        # Truncate string values in oldest outputs first
        for agent_name in seen_agents:
            if tokens_to_cut <= 0:
                break

            if agent_name not in self.agent_outputs:
                continue

            output = self.agent_outputs[agent_name]
            for key, value in list(output.items()):
                if isinstance(value, str) and len(value) > 100:
                    # Calculate how much to truncate from this value
                    value_tokens = estimate_tokens(value)
                    if value_tokens > 50:
                        # Keep at least some content
                        chars_to_keep = max(50 * CHARS_PER_TOKEN, len(value) // 4)
                        truncated = value[:chars_to_keep] + "... [truncated]"
                        output[key] = truncated
                        tokens_cut = estimate_tokens(value) - estimate_tokens(truncated)
                        tokens_to_cut -= tokens_cut

            if tokens_to_cut <= 0:
                break

        return self.estimate_context_tokens()

    def _trim_summarize(self, max_tokens: int, provider: AgentProvider) -> int:
        """Trim by summarizing context with LLM.

        Uses the provider to generate a summary of older context,
        replacing detailed outputs with a condensed summary.

        Note: This is a simplified implementation. A full implementation
        would need to be async and actually call the provider.

        Args:
            max_tokens: Maximum number of tokens allowed.
            provider: Provider to use for summarization.

        Returns:
            Number of tokens after trimming.
        """
        # Get unique agents in execution order
        seen_agents: list[str] = []
        for agent in self.execution_history:
            if agent not in seen_agents:
                seen_agents.append(agent)

        # For simplicity, we'll use drop_oldest as a fallback
        # A real implementation would call the provider to summarize
        # the oldest outputs before dropping them.

        # Summarize strategy: keep recent outputs, summarize/drop old ones
        # Keep the most recent half of agents
        recent_count = max(1, len(seen_agents) // 2)

        # For agents we're dropping, create a summary entry
        dropped_agents = seen_agents[:-recent_count] if recent_count < len(seen_agents) else []

        if dropped_agents:
            # Create a summary of dropped agents
            summary_parts = []
            for agent_name in dropped_agents:
                if agent_name in self.agent_outputs:
                    output = self.agent_outputs[agent_name]
                    # Create a brief summary of the output
                    summary = f"{agent_name}: "
                    for key, value in output.items():
                        if isinstance(value, str):
                            summary += f"{key}={value[:50]}... "
                        else:
                            summary += f"{key}={value} "
                    summary_parts.append(summary.strip())
                    del self.agent_outputs[agent_name]

            # Store summary as a special context entry
            if summary_parts:
                self.agent_outputs["_context_summary"] = {
                    "summary": "; ".join(summary_parts)[:500],
                    "dropped_agents": dropped_agents,
                }

        return self.estimate_context_tokens()
