"""Copilot Conductor - A CLI tool for defining and running multi-agent workflows.

Copilot Conductor enables orchestration of multi-agent workflows defined in YAML.
It supports conditional routing, loop-back patterns, human-in-the-loop gates,
and integrates with the GitHub Copilot SDK.

Example:
    Run a workflow from the command line::

        $ conductor run workflow.yaml --input question="What is Python?"

    Or use the library programmatically::

        from copilot_conductor.config.loader import load_config
        from copilot_conductor.engine.workflow import WorkflowEngine
        from copilot_conductor.providers.factory import create_provider

        config = load_config("workflow.yaml")
        provider = await create_provider("copilot")
        engine = WorkflowEngine(config, provider)
        result = await engine.run({"question": "What is Python?"})

Modules:
    config: Configuration loading, schema validation, and environment variable resolution.
    engine: Workflow execution engine, context management, routing, and limits.
    executor: Agent execution, template rendering, and output validation.
    providers: SDK provider abstraction and implementations.
    gates: Human-in-the-loop gate handling.
    cli: Command-line interface commands.
    exceptions: Custom exception hierarchy.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
