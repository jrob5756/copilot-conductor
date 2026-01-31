# Copilot Conductor

A CLI tool for defining and running multi-agent workflows with the GitHub Copilot SDK.

[![CI](https://github.com/your-org/copilot-conductor/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/copilot-conductor/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/copilot-conductor.svg)](https://badge.fury.io/py/copilot-conductor)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Features

- **YAML-based workflow definitions** - Define multi-agent workflows in simple, readable YAML
- **Parallel agent execution** - Run independent agents concurrently for faster workflows
- **Conditional routing** - Route between agents based on output conditions
- **Loop-back patterns** - Iterate agents until conditions are met
- **Human-in-the-loop gates** - Pause workflows for human decisions with Rich terminal UI
- **Context accumulation** - Three modes for managing context between agents
- **Safety limits** - Max iterations and timeout enforcement
- **Tool support** - Configure tools at workflow and agent levels
- **Validation** - Validate workflows before execution
- **Dry-run mode** - Preview execution plans without running

## Installation

### Using uvx (Recommended)

```bash
uvx copilot-conductor run workflow.yaml
```

### Using pipx

```bash
pipx install copilot-conductor
conductor run workflow.yaml
```

### Using pip

```bash
pip install copilot-conductor
conductor run workflow.yaml
```

## Quick Start

### 1. Create a workflow file

```yaml
# my-workflow.yaml
workflow:
  name: simple-qa
  description: A simple question-answering workflow
  entry_point: answerer

agents:
  - name: answerer
    model: gpt-4
    prompt: |
      Answer the following question:
      {{ workflow.input.question }}
    output:
      answer:
        type: string
    routes:
      - to: $end

output:
  answer: "{{ answerer.output.answer }}"
```

### 2. Run the workflow

```bash
conductor run my-workflow.yaml --input question="What is Python?"
```

### 3. View the output

```json
{
  "answer": "Python is a high-level, interpreted programming language..."
}
```

## CLI Reference

### `conductor run`

Execute a workflow from a YAML file.

```bash
conductor run <workflow.yaml> [OPTIONS]
```

**Options:**
- `--input`, `-i` `NAME=VALUE` - Workflow inputs (can be repeated)
- `--input.NAME=VALUE` - Alternative input syntax
- `--provider`, `-p` `PROVIDER` - Override the provider (default: copilot)
- `--dry-run` - Show execution plan without running
- `--skip-gates` - Auto-select first option at human gates
- `--verbose`, `-V` - Show detailed execution progress

**Examples:**

```bash
# Basic run with inputs
conductor run workflow.yaml --input question="What is AI?"

# Multiple inputs
conductor run workflow.yaml -i question="Hello" -i context="Greeting"

# Dry run to preview execution
conductor run workflow.yaml --dry-run

# Skip human gates for automation
conductor run workflow.yaml --skip-gates
```

### `conductor validate`

Validate a workflow file without executing it.

```bash
conductor validate <workflow.yaml>
```

**Example:**

```bash
conductor validate my-workflow.yaml
```

### `conductor init`

Create a new workflow file from a template.

```bash
conductor init <name> [OPTIONS]
```

**Options:**
- `--template`, `-t` `TEMPLATE` - Template to use (default: simple)
- `--output`, `-o` `PATH` - Output file path

**Examples:**

```bash
# Create a simple workflow
conductor init my-workflow

# Create from a specific template
conductor init my-workflow --template loop

# Specify output path
conductor init my-workflow -t human-gate -o ./workflows/review.yaml
```

### `conductor templates`

List available workflow templates.

```bash
conductor templates
```

## Workflow YAML Schema

### Top-level structure

```yaml
workflow:
  name: string                    # Required: Workflow identifier
  description: string             # Optional: Human-readable description
  version: string                 # Optional: Semantic version
  entry_point: string             # Required: First agent to execute
  runtime:
    provider: copilot             # Provider: copilot, openai-agents, claude
    default_model: string         # Default model for agents
  input:                          # Workflow input definitions
    param_name:
      type: string|number|boolean|array|object
      required: true              # Default: true
      default: value              # Optional default value
  context:
    mode: accumulate              # accumulate, last_only, explicit
    max_tokens: 4000              # Optional token limit
    trim_strategy: drop_oldest    # truncate, drop_oldest, summarize
  limits:
    max_iterations: 10            # Default: 10, max: 100
    timeout_seconds: 600          # Default: 600, max: 3600
  hooks:
    on_start: "{{ template }}"
    on_complete: "{{ template }}"
    on_error: "{{ template }}"

tools:                            # Workflow-level tools
  - tool_name

parallel:                         # Parallel execution groups
  - name: string                  # Required: Group identifier
    description: string           # Optional: Purpose description
    agents:                       # Required: Agents to run in parallel
      - agent_name_1
      - agent_name_2
    failure_mode: fail_fast       # fail_fast|continue_on_error|all_or_nothing
    routes:                       # Routes after parallel execution
      - to: agent_name|$end
        when: "{{ condition }}"

agents:
  - name: string                  # Required: Agent identifier
    type: agent|human_gate        # Default: agent
    model: string                 # Model to use
    input:                        # Context dependencies
      - workflow.input.param
      - other_agent.output
      - optional_agent.output?    # Optional with ?
    tools:                        # null=all, []=none, [list]=subset
      - tool_name
    system_prompt: string         # System message
    prompt: string                # User prompt (Jinja2 template)
    output:                       # Output schema
      field_name:
        type: string|number|boolean|array|object
    routes:
      - to: agent_name|$end
        when: "{{ condition }}"   # Optional condition
        output:                   # Optional output transform
          key: "{{ template }}"
    options:                      # For human_gate type
      - label: string
        value: string
        route: agent_name|$end
        prompt_for: field_name    # Optional text input

output:                           # Final output templates
  key: "{{ template }}"
```

### Context Modes

- **accumulate** (default): All prior agent outputs available to subsequent agents
- **last_only**: Only the most recent agent's output available
- **explicit**: Only inputs listed in the agent's `input` array are available

### Routing

Routes are evaluated in order; first matching `when` condition wins. A route without `when` always matches.

```yaml
routes:
  - to: handler
    when: "{{ output.score > 7 }}"
  - to: fallback
    when: "score < 3"              # simpleeval syntax also supported
  - to: $end                       # Default route
```

## Examples

See the [`examples/`](./examples/) directory for complete workflow examples:

- [`simple-qa.yaml`](./examples/simple-qa.yaml) - Simple question-answering workflow
- [`parallel-research.yaml`](./examples/parallel-research.yaml) - Parallel research from multiple sources
- [`parallel-validation.yaml`](./examples/parallel-validation.yaml) - Parallel code validation checks
- [`design-review.yaml`](./examples/design-review.yaml) - Loop pattern with human gate
- [`research-assistant.yaml`](./examples/research-assistant.yaml) - Multi-agent workflow with tools
- [`design.yaml`](./examples/design.yaml) - Solution design workflow with architect/reviewer loop
- [`plan.yaml`](./examples/plan.yaml) - Implementation planning workflow with quality gates

For detailed information on parallel execution, see [Parallel Execution Guide](./docs/parallel-execution.md).

## Development

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/copilot-conductor.git
cd copilot-conductor

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --group dev
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific tests
uv run pytest tests/test_engine/
```

### Linting and Type Checking

```bash
# Run ruff linter
uv run ruff check .

# Run ruff with auto-fix
uv run ruff check . --fix

# Run type checker
uv run mypy src/
```

### Code Style

This project uses:
- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- [MyPy](https://mypy.readthedocs.io/) for type checking
- Google-style docstrings

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`uv run pytest && uv run ruff check .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see [LICENSE](./LICENSE) for details.
