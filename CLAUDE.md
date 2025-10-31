# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`any-agent` is a unified interface for using and evaluating different agent frameworks. It provides a single API (`AnyAgent`) that wraps multiple agent frameworks (TinyAgent, Google ADK, LangChain, LlamaIndex, OpenAI Agents, Smolagents, Agno) with consistent configuration, tracing, callbacks, and tool integration.

Key concepts:
- **AnyAgent**: Abstract base class providing unified interface across frameworks
- **AgentConfig**: Configuration for agents (model, instructions, tools, callbacks)
- **AgentTrace**: OpenTelemetry-based tracing system that captures agent execution spans
- **Framework-specific implementations**: Each framework has a wrapper in `src/any_agent/frameworks/`
- **Tool wrapping**: Tools are normalized across frameworks via `_wrap_tools()` in `src/any_agent/tools/wrappers.py`
- **Callback system**: Unified callback hooks that work across all frameworks via wrappers in `src/any_agent/callbacks/wrappers/`

## Development Setup

This project uses `uv` for dependency management:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install with all framework dependencies
uv sync --dev --extra all

# Install with specific frameworks only (e.g., langchain and openai)
uv sync --dev --extra langchain --extra openai
```

## Common Commands

### Testing

```bash
# Run all tests
pytest -v tests

# Run unit tests only
pytest -v tests/unit

# Run integration tests only
pytest -v tests/integration

# Run tests in parallel
pytest -v tests -n auto

# Run specific test file
pytest -v tests/unit/test_config.py

# Run with coverage
pytest -v tests --cov=src/any_agent --cov-report=html
```

### Linting

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run specific checks
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

### Documentation

```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## Architecture

### Framework Pattern

Each agent framework follows the same integration pattern:

1. **Framework-specific class** in `src/any_agent/frameworks/<framework>.py`:
   - Inherits from `AnyAgent`
   - Implements `_load_agent()` to initialize the framework's agent
   - Implements `_run_async()` to execute the agent
   - Implements `update_output_type_async()` for structured outputs

2. **Callback wrapper** in `src/any_agent/callbacks/wrappers/<framework>.py`:
   - Wraps framework-specific methods to inject callbacks
   - Manages callback context for tracing

3. **Span generation** in `src/any_agent/callbacks/span_generation/<framework>.py`:
   - Creates OpenTelemetry spans from framework-specific execution traces

### Tracing System

The tracing system uses OpenTelemetry to capture execution:
- `AgentTrace` stores all spans and final output
- Each `run_async()` creates a root span
- Framework callbacks generate child spans for tool calls and LLM interactions
- Spans include token counts, timing, and metadata
- On error, `AgentRunError` wraps the exception and includes the trace

### Tool System

Tools can be:
1. **Python functions**: Wrapped directly
2. **MCP (Model Context Protocol)**: Connected via `MCPClient` and translated to framework format
3. **A2A (Agent-to-Agent)**: Other agents exposed as tools via `a2a_tool()`

Tool wrapping happens in `_wrap_tools()` which:
- Identifies MCP tools and creates `MCPClient` instances
- Converts tools to framework-specific format
- Returns both wrapped tools and MCP clients (for cleanup)

### Async Pattern

The codebase uses async throughout with a sync wrapper:
- Core methods are async (`create_async`, `run_async`, `serve_async`)
- Sync methods (`create`, `run`) use `run_async_in_sync()`
- Jupyter notebook support via `INSIDE_NOTEBOOK` detection and `nest_asyncio`

## Testing Notes

- Test configuration in `pyproject.toml`: `asyncio_mode = "auto"`, 180s timeout
- Integration tests require API keys for various providers (set via environment variables)
- Use `pytest-xdist` for parallel execution
- Snapshot testing via `syrupy` for trace validation
- Coverage excludes `src/any_agent/vendor/` and `tests/`

## Type Checking

- Strict mypy configuration: `strict = true`
- Use type hints everywhere
- Test files have relaxed decorator typing due to pytest decorators

## Code Style

- Ruff for linting (extends ALL rules, then ignores specific ones)
- Key ignored rules: D100 (module docstrings), D104 (package docstrings), ANN (annotations handled by mypy)
- Per-file ignores in `pyproject.toml` for specific modules
- All new code must pass `pre-commit run --all-files`
