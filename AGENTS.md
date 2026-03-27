# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`CLAUDE.md` is a symlink to this file. Always edit `AGENTS.md` directly; never modify `CLAUDE.md`.

## Where to Look First

- [README.md](README.md): high-level usage and supported frameworks.
- [CONTRIBUTING.md](CONTRIBUTING.md): canonical dev setup, test matrix, and contribution workflow.
- [pyproject.toml](pyproject.toml) and [.pre-commit-config.yaml](.pre-commit-config.yaml): formatting/lint/typecheck configuration.
- [docs/](docs/): MkDocs site sources (configured by [mkdocs.yml](mkdocs.yml)).

## Project Structure & Module Organization

- `src/any_agent/`: Core library. `AnyAgent` is the main entry point; `AgentConfig` and `AgentFramework` define config.
- `src/any_agent/frameworks/`: One module per supported framework (agno, google, langchain, llama_index, openai, smolagents, tinyagent). Each implements `AnyAgent`'s abstract methods via lazy imports to keep optional dependencies isolated.
- `src/any_agent/callbacks/`: Callback system with per-framework wrappers (`wrappers/`) and span generation (`span_generation/`). Base class is `Callback` in `base.py`.
- `src/any_agent/tools/`: Built-in tools (web browsing, user interaction, final output) plus MCP client integration (`mcp/`), A2A tools, and Composio integration.
- `src/any_agent/tracing/`: OpenTelemetry-based tracing. `AgentTrace` / `AgentSpan` are the primary types.
- `src/any_agent/evaluation/`: LLM judge and agent judge evaluators.
- `src/any_agent/serving/`: A2A protocol and MCP server implementations for serving agents.
- `src/any_agent/vendor/`: Vendored adapters (langchain_any_llm, llama_index_utils).
- `tests/`: `unit/`, `integration/`, `snapshots/`, plus shared fixtures in `tests/conftest.py`.

## Build, Test, and Development Commands

This repo uses `uv` for local dev (Python 3.11+). For the full, up-to-date command set, follow [CONTRIBUTING.md](CONTRIBUTING.md).

- Create env + install dev deps: `uv venv && source .venv/bin/activate && uv sync --dev --extra all`
- Run all checks (preferred): `uv run pre-commit run --all-files --verbose`
- Unit tests: `uv run pytest -v tests/unit`
- Single test file: `uv run pytest -v tests/unit/frameworks/test_tinyagent.py`
- Single test: `uv run pytest -v tests/unit/frameworks/test_tinyagent.py::test_function_name`
- Integration tests (require API keys): `uv run pytest -v tests/integration`
- Docs preview: `uv run mkdocs serve`

## Coding Style & Naming Conventions

- Python indentation: 4 spaces; formatting/linting via `ruff` and `pre-commit`. Line length is not explicitly capped (E501 is ignored).
- Type hints: required; `mypy` runs in strict mode for library code (see `pyproject.toml`). The mypy pre-commit hook runs via `uv run mypy`.
- Framework code lives under `src/any_agent/frameworks/<framework>.py` (keep framework-specific behavior isolated there).
- Lazy imports are used extensively for optional framework dependencies (e.g., `smolagents`, `langchain`, `openai-agents`). The ruff rule `PLC0415` is disabled to allow this.
- Prefer direct attribute access (e.g., `obj.field`) over `getattr(obj, "field")` when the field is typed.
- Please add code comments if you find them helpful to accomplish your objective. However, please remove any comments you added that describe obvious behavior before finishing your task.
- Never use emdashes or -- in any comments or descriptions.

## Testing Guidelines

- Framework: `pytest` (+ `pytest-asyncio`, `pytest-xdist`). Async mode is `auto` with session-scoped event loop.
- Add/adjust tests with every change (happy path + error cases). Integration tests should `pytest.skip(...)` when credentials/services aren't available.
- New code should target ~85%+ coverage. Write tests for every branch in new code, including error/raise paths and edge cases, so that patch coverage passes in CI.
- Do not use class-based test grouping (`class TestFoo:`). All tests should be standalone functions.
- Do not add decorative section-separator comments (e.g., `# -----------` banners).
- Place imports at the top of test files unless the import is for an optional dependency that may not be installed (e.g., framework-specific SDKs). In that case, inline imports inside the test function are acceptable.
- Snapshot tests live in `tests/snapshots/` using `syrupy`.

## Architecture Notes

- `AnyAgent.create()` / `create_async()` is the main factory. It resolves `AgentFramework` to a concrete subclass via `_get_agent_type_by_framework()`, instantiates it, then calls `_load_agent()`.
- Every framework subclass must implement `_load_agent()`, `_run_async()`, `update_output_type_async()`, and the `framework` property.
- Tools are normalized to framework-native types by `_wrap_tools()` in `tools/wrappers.py`. MCP tools (stdio, SSE, streamable HTTP) are connected via `MCPClient`.
- The callback system hooks into each framework differently via `callbacks/wrappers/<framework>.py`. Wrappers monkey-patch or register framework-native hooks to fire the unified `Callback` methods.
- `AgentTrace` collects `AgentSpan` objects (OpenTelemetry spans). Traces include token/cost info, duration, and can be serialized to JSON.
- `AgentCancel` is a special exception base class for intentional cancellation from callbacks. It propagates through framework error wrapping via `_unwrap_agent_cancel()`.
- The `any-llm-sdk` package (from the sibling `any-llm` repo) is a core dependency providing the unified LLM interface used by all framework integrations.

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits: `feat(scope): ...`, `fix: ...`, `chore(deps): ...`, `tests: ...`.
- PRs should include a clear description, linked issues, and completed checklist.

## Security & Configuration Tips

- Never commit secrets. Use environment variables or a local `.env` (gitignored) for provider API keys.
