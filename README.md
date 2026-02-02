# Conductor

A CLI tool for defining and running multi-agent workflows with the GitHub Copilot SDK and Anthropic Claude.

[![CI](https://github.com/microsoft/conductor/actions/workflows/ci.yml/badge.svg)](https://github.com/microsoft/conductor/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/conductor-cli.svg)](https://badge.fury.io/py/conductor-cli)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Why Conductor?

A single LLM prompt can answer a question, but it can't review its own work, research from multiple angles, or pause for human approval. You need multi-agent workflows—but building them means coding custom solutions, managing state, handling failures, and hoping you don't create infinite loops.

Conductor provides the patterns that work: evaluator-optimizer loops for iterative refinement, parallel execution with failure modes, and human-in-the-loop gates. Define them in YAML with built-in safety limits. Version control your workflows like code.

## Features

- **YAML-based workflows** - Define multi-agent workflows in readable YAML
- **Multiple providers** - GitHub Copilot or Anthropic Claude with seamless switching
- **Parallel execution** - Run agents concurrently (static groups or dynamic for-each)
- **Conditional routing** - Route between agents based on output conditions
- **Human-in-the-loop** - Pause for human decisions with Rich terminal UI
- **Safety limits** - Max iterations and timeout enforcement
- **Validation** - Validate workflows before execution

## Installation

### Using uv (Recommended)

```bash
# Run without installing (use --from since package name differs from command)
uvx --from conductor-cli conductor run workflow.yaml

# Or install persistently
uv tool install conductor-cli
conductor run workflow.yaml
```

### Using pipx

```bash
pipx install conductor-cli
conductor run workflow.yaml
```

### Using pip

```bash
pip install conductor-cli
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

## Providers

Conductor supports multiple AI providers. Choose based on your needs:

| Feature | Copilot | Claude |
|---------|---------|--------|
| **Pricing** | Subscription ($10-39/mo) | Pay-per-token |
| **Context Window** | 8K-128K tokens | 200K tokens |
| **Tool Support (MCP)** | Yes | Planned |
| **Streaming** | Yes | Planned |
| **Best For** | Heavy usage, tools | Large context, pay-per-use |

### Using Claude

```yaml
workflow:
  runtime:
    provider: claude
    default_model: claude-sonnet-4.5-latest
```

Set your API key: `export ANTHROPIC_API_KEY=sk-ant-...`

**See also:** [Claude Documentation](docs/providers/claude.md) | [Provider Comparison](docs/providers/comparison.md) | [Migration Guide](docs/providers/migration.md)

## CLI Reference

### `conductor run`

Execute a workflow from a YAML file.

```bash
conductor run <workflow.yaml> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-i, --input NAME=VALUE` | Workflow input (repeatable) |
| `-p, --provider PROVIDER` | Override provider |
| `--dry-run` | Preview execution plan |
| `--skip-gates` | Auto-select at human gates |
| `-V, --verbose` | Show detailed progress |

### `conductor validate`

Validate a workflow file without executing.

```bash
conductor validate <workflow.yaml>
```

### `conductor init`

Create a new workflow from a template.

```bash
conductor init <name> --template <template> --output <path>
```

### `conductor templates`

List available workflow templates.

```bash
conductor templates
```

**Full CLI documentation:** [docs/cli-reference.md](docs/cli-reference.md)

## Examples

See the [`examples/`](./examples/) directory for complete workflows:

| Example | Description |
|---------|-------------|
| [simple-qa.yaml](./examples/simple-qa.yaml) | Basic single-agent Q&A |
| [for-each-simple.yaml](./examples/for-each-simple.yaml) | Dynamic parallel processing |
| [parallel-research.yaml](./examples/parallel-research.yaml) | Static parallel execution |
| [design-review.yaml](./examples/design-review.yaml) | Human gate with loop pattern |
| [kpi-analysis-parallel.yaml](./examples/kpi-analysis-parallel.yaml) | For-each with parallel analysis |

**More examples and running instructions:** [examples/README.md](./examples/README.md)

## Documentation

| Document | Description |
|----------|-------------|
| [Workflow Syntax](./docs/workflow-syntax.md) | Complete YAML schema reference |
| [CLI Reference](./docs/cli-reference.md) | Full command-line documentation |
| [Parallel Execution](./docs/parallel-execution.md) | Static parallel groups |
| [Dynamic Parallel](./docs/dynamic-parallel.md) | For-each groups and array processing |
| [Claude Provider](./docs/providers/claude.md) | Claude setup and configuration |
| [Provider Comparison](./docs/providers/comparison.md) | Copilot vs Claude decision guide |

## Development

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setup

```bash
git clone https://github.com/microsoft/conductor.git
cd conductor
make dev
```

### Common Commands

```bash
make test             # Run tests
make test-cov         # Run tests with coverage
make lint             # Check linting
make format           # Auto-fix and format code
make typecheck        # Type check
make check            # Run all checks (lint + typecheck)
make validate-examples  # Validate all example workflows
```

### Code Style

- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- [ty](https://github.com/astral-sh/ty) for type checking
- Google-style docstrings

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and checks (`make test && make check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see [LICENSE](./LICENSE) for details.
