# Conductor

A CLI tool for defining and running multi-agent workflows with the GitHub Copilot SDK and Anthropic Claude.

[![CI](https://github.com/your-org/conductor/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/conductor/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/conductor.svg)](https://badge.fury.io/py/conductor)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Features

- **YAML-based workflow definitions** - Define multi-agent workflows in simple, readable YAML
- **Multiple AI providers** - GitHub Copilot or Anthropic Claude with seamless switching
- **Static parallel execution** - Run independent agents concurrently for faster workflows
- **Dynamic parallel (for-each)** - Process variable-length arrays with parallel agent instances
- **Conditional routing** - Route between agents based on output conditions
- **Loop-back patterns** - Iterate agents until conditions are met
- **Human-in-the-loop gates** - Pause workflows for human decisions with Rich terminal UI
- **Context accumulation** - Three modes for managing context between agents
- **Safety limits** - Max iterations and timeout enforcement
- **Tool support** - Configure tools at workflow and agent levels (Copilot provider)
- **Validation** - Validate workflows before execution
- **Dry-run mode** - Preview execution plans without running

## Installation

### Using uvx (Recommended)

```bash
uvx conductor run workflow.yaml
```

### Using pipx

```bash
pipx install conductor
conductor run workflow.yaml
```

### Using pip

```bash
pip install conductor
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
    model: gpt-5.2
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

## Using Claude Provider

Conductor supports Anthropic Claude models via the official Anthropic SDK.

### 1. Install the Claude SDK

```bash
uv add 'anthropic>=0.77.0,<1.0.0'
# or
pip install 'anthropic>=0.77.0,<1.0.0'
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

Get your API key at [console.anthropic.com](https://console.anthropic.com)

### 3. Update your workflow

```yaml
# my-claude-workflow.yaml
workflow:
  name: simple-qa
  description: A simple question-answering workflow with Claude
  entry_point: answerer
  runtime:
    provider: claude  # Change from 'copilot' to 'claude'
    default_model: claude-sonnet-4.5-latest
    temperature: 0.7
    max_tokens: 2048  # Note: max_tokens not max_tokens_claude

agents:
  - name: answerer
    model: claude-sonnet-4.5-latest
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

### 4. Run with Claude

```bash
conductor run my-claude-workflow.yaml --input question="What is Python?"
```

### Available Claude Models

| Model | Best For | Cost |
|-------|----------|------|
| `claude-sonnet-4.5-latest` | General purpose (recommended) | $3/$15 per MTok |
| `claude-haiku-4.5-latest` | Fast, simple tasks | $1/$5 per MTok |
| `claude-opus-4.5-latest` | Complex reasoning | $5/$25 per MTok |

**Pricing**: Input/Output per 1M tokens. [Verify latest pricing](https://www.anthropic.com/pricing)

### Claude vs Copilot

| Feature | Copilot | Claude |
|---------|---------|--------|
| Pricing | Subscription ($10-39/mo) | Pay-per-token |
| Context | 8K-128K tokens | 200K tokens |
| Tools (MCP) | ✅ Yes | ⏳ Phase 2+ |
| Streaming | ✅ Yes | ⏳ Phase 2+ |

**See also**: [Provider Comparison](docs/providers/comparison.md) | [Migration Guide](docs/providers/migration.md) | [Claude Documentation](docs/providers/claude.md)

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

parallel:                         # Static parallel execution groups
  - name: string                  # Required: Group identifier
    description: string           # Optional: Purpose description
    agents:                       # Required: Agents to run in parallel
      - agent_name_1
      - agent_name_2
    failure_mode: fail_fast       # fail_fast|continue_on_error|all_or_nothing
    routes:                       # Routes after parallel execution
      - to: agent_name|$end
        when: "{{ condition }}"

for_each:                         # Dynamic parallel (for-each) groups
  - name: string                  # Required: Group identifier
    type: for_each                # Required: Marks as dynamic parallel
    description: string           # Optional: Purpose description
    source: string                # Required: Reference to array (e.g., "finder.output.items")
    as: string                    # Required: Loop variable name (e.g., "item")
    max_concurrent: 10            # Optional: Concurrent execution limit (default: 10)
    failure_mode: fail_fast       # Optional: fail_fast|continue_on_error|all_or_nothing
    key_by: string                # Optional: Path to extract key (e.g., "item.id")
    agent:                        # Required: Inline agent definition
      model: string
      prompt: "{{ item }}"        # Can use loop variables
      output:
        field: { type: string }
    routes:                       # Routes after for-each execution
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
- [`for-each-simple.yaml`](./examples/for-each-simple.yaml) - **Dynamic parallel (for-each) processing**
- [`kpi-analysis-parallel.yaml`](./examples/kpi-analysis-parallel.yaml) - **Parallel KPI analysis with for-each**
- [`parallel-research.yaml`](./examples/parallel-research.yaml) - Parallel research from multiple sources
- [`parallel-validation.yaml`](./examples/parallel-validation.yaml) - Parallel code validation checks
- [`design-review.yaml`](./examples/design-review.yaml) - Loop pattern with human gate
- [`research-assistant.yaml`](./examples/research-assistant.yaml) - Multi-agent workflow with tools
- [`design.yaml`](./examples/design.yaml) - Solution design workflow with architect/reviewer loop
- [`plan.yaml`](./examples/plan.yaml) - Implementation planning workflow with quality gates

**Documentation:**
- [Parallel Execution Guide](./docs/parallel-execution.md) - Static parallel groups
- [Dynamic Parallel (For-Each) Guide](./docs/dynamic-parallel.md) - For-each groups and array processing
- [Workflow Syntax Reference](./docs/workflow-syntax.md) - Complete YAML syntax

## Development

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/conductor.git
cd conductor

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
