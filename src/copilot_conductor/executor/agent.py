"""Agent execution orchestration for Copilot Conductor.

This module provides the AgentExecutor class for executing a single agent
with prompt rendering and output validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from copilot_conductor.executor.output import parse_json_output, validate_output
from copilot_conductor.executor.template import TemplateRenderer
from copilot_conductor.providers.base import AgentOutput

if TYPE_CHECKING:
    from copilot_conductor.config.schema import AgentDef
    from copilot_conductor.providers.base import AgentProvider


class AgentExecutor:
    """Executes a single agent with prompt rendering and output validation.

    The AgentExecutor handles the complete lifecycle of executing an agent:
    1. Render the prompt template with the provided context
    2. Execute the agent via the provider
    3. Validate the output against the agent's schema (if defined)

    Example:
        >>> from copilot_conductor.providers.copilot import CopilotProvider
        >>> provider = CopilotProvider()
        >>> executor = AgentExecutor(provider)
        >>> output = await executor.execute(agent, context)
    """

    def __init__(self, provider: AgentProvider) -> None:
        """Initialize the AgentExecutor.

        Args:
            provider: The agent provider to use for execution.
        """
        self.provider = provider
        self.renderer = TemplateRenderer()

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
    ) -> AgentOutput:
        """Execute an agent with the given context.

        This method:
        1. Renders the agent's prompt template with context
        2. Calls the provider to execute the agent
        3. Validates output against the agent's schema (if defined)

        Args:
            agent: Agent definition from workflow config.
            context: Context for prompt rendering, built by WorkflowContext.

        Returns:
            Validated agent output.

        Raises:
            TemplateError: If prompt rendering fails.
            ProviderError: If agent execution fails.
            ValidationError: If output doesn't match schema.
        """
        # Render prompt with context
        rendered_prompt = self.renderer.render(agent.prompt, context)

        # Render system prompt if present (used by some providers)
        # Note: System prompt support will be fully utilized in later EPICs
        if agent.system_prompt:
            _ = self.renderer.render(agent.system_prompt, context)

        # Execute via provider
        output = await self.provider.execute(
            agent=agent,
            context=context,
            rendered_prompt=rendered_prompt,
            tools=agent.tools,
        )

        # Ensure output.content is a dict
        if not isinstance(output.content, dict):
            # Try to parse raw response as JSON if content is not a dict
            if output.raw_response and isinstance(output.raw_response, str):
                output = AgentOutput(
                    content=parse_json_output(output.raw_response),
                    raw_response=output.raw_response,
                    tokens_used=output.tokens_used,
                    model=output.model,
                )
            else:
                # Wrap the content in a dict
                output = AgentOutput(
                    content={"result": output.content},
                    raw_response=output.raw_response,
                    tokens_used=output.tokens_used,
                    model=output.model,
                )

        # Validate output against schema
        if agent.output:
            validate_output(output.content, agent.output)

        return output

    def render_prompt(self, agent: AgentDef, context: dict[str, Any]) -> str:
        """Render an agent's prompt template.

        This is useful for debugging or dry-run mode.

        Args:
            agent: Agent definition from workflow config.
            context: Context for prompt rendering.

        Returns:
            Rendered prompt string.

        Raises:
            TemplateError: If prompt rendering fails.
        """
        return self.renderer.render(agent.prompt, context)
