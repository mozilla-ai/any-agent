# Agent Development Guidelines

## Setup
```bash
uv venv && source .venv/bin/activate && uv sync --dev --extra all
```

## Commands
- **Lint**: `pre-commit run --all-files` (runs ruff, mypy, codespell)
- **Test all**: `pytest -v tests`
- **Test single file**: `pytest -v tests/path/to/test_file.py`
- **Test single function**: `pytest -v tests/path/to/test_file.py::test_function_name`
- **Docs**: `mkdocs serve`

## Code Style
- **Imports**: Use `from __future__ import annotations` at top; lazy imports (`import` inside functions) allowed via PLC0415
- **Types**: Strict mypy enabled—all functions must have type hints; use `Optional[T]` not `T | None` for Google ADK compatibility (see src/any_agent/tools/a2a.py:68-71)
- **Formatting**: Ruff handles formatting; 88 char line length preferred but not enforced (E501 ignored)
- **Docstrings**: Required for classes/functions (D101, D103); use Google style; modules/packages don't need them (D100, D104 ignored)
- **Error handling**: Define custom exceptions (see `AgentRunError`); document failure modes; no bare except (BLE001 ignored but use judiciously)
- **Naming**: Snake_case for functions/vars; PascalCase for classes
- **Comments**: Code must be self-documenting; add comments only for non-obvious logic

## PR Requirements
- All tests must pass; no failing/untested code
- PRs must be <500 LOC and focused on one purpose
- Include unit tests for new functionality (happy path + edge cases)
