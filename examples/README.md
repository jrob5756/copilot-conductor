# Copilot Conductor Examples

This directory contains example workflow files demonstrating various features of Copilot Conductor.

## Examples

### simple-qa.yaml

A minimal workflow with a single agent that answers questions. Demonstrates:
- Basic workflow structure
- Input parameters
- Output schema validation
- Simple routing to `$end`

```bash
conductor run examples/simple-qa.yaml --input question="What is Python?"
```

### design-review.yaml

An iterative design workflow with human-in-the-loop approval. Demonstrates:
- Multiple agents with conditional routing
- Loop-back patterns for refinement
- Human gates for approval decisions
- Context accumulation between iterations
- Safety limits (max_iterations, timeout)

```bash
# Interactive mode
conductor run examples/design-review.yaml --input requirement="Build a REST API"

# Automation mode (auto-approves)
conductor run examples/design-review.yaml --input requirement="Build a REST API" --skip-gates
```

### research-assistant.yaml

A multi-agent research workflow with tools. Demonstrates:
- Multiple specialized agents
- Tool configuration at workflow and agent levels
- Explicit context mode
- Conditional routing based on coverage
- Complex output schemas

```bash
conductor run examples/research-assistant.yaml --input topic="AI in healthcare"

# With custom depth
conductor run examples/research-assistant.yaml --input topic="Quantum computing" --input depth="comprehensive"
```

## Running Examples

### Prerequisites

1. Install Copilot Conductor:
   ```bash
   uvx copilot-conductor
   ```

2. Ensure you have valid credentials for the Copilot SDK (or the provider you're using).

### Validate Before Running

You can validate a workflow without executing it:

```bash
conductor validate examples/simple-qa.yaml
```

### Dry Run

Preview the execution plan without actually running the workflow:

```bash
conductor run examples/simple-qa.yaml --dry-run
```

### Verbose Mode

See detailed execution progress:

```bash
conductor run examples/simple-qa.yaml --input question="Hello" --verbose
```

## Creating Your Own Workflows

Use the `init` command to create a new workflow from a template:

```bash
# List available templates
conductor templates

# Create from a template
conductor init my-workflow --template loop
```

## Tips

1. **Start simple**: Begin with a linear workflow and add complexity incrementally.

2. **Validate often**: Use `conductor validate` to catch configuration errors early.

3. **Use dry-run**: Preview execution with `--dry-run` before running expensive workflows.

4. **Explicit context**: Use `context.mode: explicit` for complex workflows to control exactly what context each agent sees.

5. **Safety limits**: Always set appropriate `max_iterations` and `timeout_seconds` for workflows with loops.

6. **Optional dependencies**: Use `?` suffix for optional input references to avoid errors when agents haven't run yet.
