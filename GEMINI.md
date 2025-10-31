# Gemini Code Assistant Context

## Project Overview

This project, `any-agent`, is a Python library that provides a unified interface for using and evaluating various agent frameworks. It allows developers to switch between different agent backends like `tinyagent`, `LangChain`, `LlamaIndex`, and others without changing their code.

The main components are:
- **`AnyAgent`**: The primary class for creating and interacting with agents.
- **`AgentConfig`**: A configuration class for defining agent properties like the model, instructions, and tools.
- **`AgentTrace`**: A class for tracing the execution of an agent.

The project is structured as a standard Python library with source code in the `src` directory and tests in the `tests` directory. It uses `pyproject.toml` for dependency management and project configuration.

## Building and Running

### Installation

To install the library and its dependencies, use `pip`:

```bash
pip install -e .
```

To install with all the optional dependencies for all supported frameworks:

```bash
pip install -e '.[all]'
```

### Running Tests

The project uses `pytest` for testing. To run the tests, use the following command:

```bash
pytest
```

### Running the Evaluator

The project includes a command-line tool for evaluating agents. To use it, run:

```bash
any-agent-evaluate --help
```

## Development Conventions

- **Linting**: The project uses `ruff` and `pylint` for linting. Configuration is in `pyproject.toml`.
- **Typing**: The project uses `mypy` for static type checking. Configuration is in `pyproject.toml`.
- **Pre-commit Hooks**: The project uses `pre-commit` to run checks before each commit. The configuration is in `.pre-commit-config.yaml`.
- **Documentation**: The documentation is built using `mkdocs` and is located in the `docs` directory.
