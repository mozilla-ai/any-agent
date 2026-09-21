# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`CLAUDE.md` is a symlink to this file. Always edit `AGENTS.md` directly; never modify `CLAUDE.md`.

## Project Status: Soft Deprecation

`any-agent` is in soft deprecation. It is still published and maintained, but:

- **Bug fixes and security fixes only.** No new features are planned.
- **New projects should use [`mozilla-ai-tinyagent`](https://github.com/mozilla-ai/tinyagent)** (PyPI: `mozilla-ai-tinyagent`) if they only need the core agent loop. That package is where the distilled agent loop now lives.
- **Reach for `any-agent`** when you specifically need to run or evaluate agents across multiple frameworks (Agno, Google ADK, LangChain, LlamaIndex, OpenAI Agents SDK, smolagents, TinyAgent) under one API.

Before starting feature work, check whether the change belongs in `tinyagent` instead. See the banner in [README.md](README.md) for the user-facing statement.

## Where to Look First

- [README.md](README.md): high-level usage and supported frameworks.
- [CONTRIBUTING.md](CONTRIBUTING.md): canonical dev setup, test matrix, and contribution workflow.
- [pyproject.toml](pyproject.toml) and [.pre-commit-config.yaml](.pre-commit-config.yaml): formatting/lint/typecheck configuration.
- [docs/](docs/): MkDocs documentation site (configured by [mkdocs.yml](mkdocs.yml)).

## Project Structure & Module Organization

- `src/any_agent/`: Core library. `AnyAgent` is the main entry point; `AgentConfig` and `AgentFramework` define config.
- `src/any_agent/frameworks/`: One module per supported framework (agno, google, langchain, llama_index, openai, smolagents, tinyagent). Each implements `AnyAgent`'s abstract methods via lazy imports to keep optional dependencies isolated. The `tinyagent` module is a thin shim that delegates to the standalone [`mozilla-ai-tinyagent`](https://github.com/mozilla-ai/tinyagent) package; the loop itself lives there.
- `src/any_agent/callbacks/`: Callback system with per-framework wrappers (`wrappers/`) and span generation (`span_generation/`). Base class is `Callback` in `base.py`.
- `src/any_agent/tools/`: Built-in tools (web browsing, user interaction, final output) plus MCP client integration (`mcp/`), A2A tools, and Composio integration.
- `src/any_agent/tracing/`: OpenTelemetry tracing setup plus attribute keys (`AnyAgentAttributes`, and `GenAI` re-exported from `tinyagent`). The `AgentTrace` / `AgentSpan` / `TokenInfo` / `CostInfo` / `AgentMessage` types are re-exported from the `tinyagent` package by `agent_trace.py`, so both packages share the same classes; `otel_types.py` re-exports the OTel value types the same way.
- `src/any_agent/evaluation/`: LLM judge and agent judge evaluators.
- `src/any_agent/serving/`: A2A protocol and MCP server implementations for serving agents.
- `src/any_agent/vendor/`: Vendored adapters (langchain_any_llm, llama_index_utils). Excluded from the ruff lint hook (`ruff-format` still applies) and from coverage.
- `src/any_agent/testing/`: Test helpers (`helpers.py`) shipped with the package.
- `src/any_agent/utils/`: Small shared utilities (`cast.py`).
- `demo/`: Standalone Streamlit demo deployed to Hugging Face Spaces. It has its own `requirements.txt` and is excluded from the ruff lint hook (`ruff-format` still applies) and from mypy; it is not part of the library.
- `scripts/`: Repo tooling (`convert_to_gitbook.py`, `generate_api_docs.py`, `wake_up_hf_endpoint.py`), excluded from mypy.
- `tests/`: `unit/`, `integration/`, `snapshots/`, `cookbooks/` (runs the notebooks under `docs/cookbook/`), `docs/` (checks code blocks in the docs via `mktestdocs`), and `assets/` (fixture data: trace JSON and HTML), plus shared fixtures in `tests/conftest.py`.

## Build, Test, and Development Commands

This repo uses `uv` for local dev (Python 3.11+). For the full, up-to-date command set, follow [CONTRIBUTING.md](CONTRIBUTING.md).

- Create env + install dev deps: `uv venv && source .venv/bin/activate && uv sync --dev --extra all`
- Run all checks (preferred): `uv run pre-commit run --all-files --verbose`
- Unit tests: `uv run pytest -v tests/unit`
- Single test file: `uv run pytest -v tests/unit/frameworks/test_tinyagent.py`
- Single test: `uv run pytest -v tests/unit/frameworks/test_tinyagent.py::test_function_name`
- Integration tests (require API keys): `uv run pytest -v tests/integration`
- Docs preview: `uv run mkdocs serve` (serves on `http://127.0.0.1:8000`)

GitBook-specific syntax (`{% hint %}`, `{% content-ref %}`) renders as raw text under `mkdocs serve`; the content is readable but unstyled. For an exact preview, push the branch and check the `gitbook-staging` deployment. The published site is built by `.github/workflows/docs.yaml`, which runs `scripts/convert_to_gitbook.py` and pushes the result to the `gitbook-docs` branch, served at <https://docs.mozilla.ai/any-agent/>.

## Coding Style & Naming Conventions

- Python indentation: 4 spaces; formatting/linting via `ruff` and `pre-commit`. Line length is not explicitly capped (E501 is ignored).
- Type hints: required; `mypy` runs in strict mode for library code (see `pyproject.toml`). The mypy pre-commit hook runs via `uv run --extra all --extra a2a --extra composio mypy`, and excludes `demo/` and `scripts/`.
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
- `AgentTrace` collects `AgentSpan` objects (OpenTelemetry spans). Traces include token/cost info, duration, and can be serialized to JSON. Both types are defined in the `tinyagent` package and re-exported here, so changes to their shape belong upstream.
- `AgentCancel` is a special exception base class for intentional cancellation from callbacks. It propagates through framework error wrapping via `_unwrap_agent_cancel()`.
- The `any-llm-sdk` package (from the sibling `any-llm` repo) is a core dependency providing the unified LLM interface used by all framework integrations.
- `mozilla-ai-tinyagent` is also a core (non-optional) dependency. Beyond the `TinyAgent` shim, `config.py` re-exports its MCP config types (`MCPParams`, `MCPSse`, `MCPStdio`, `MCPStreamableHttp`, `Tool`), so those live upstream too.

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits: `feat(scope): ...`, `fix: ...`, `chore(deps): ...`, `tests: ...`.
- PRs should include a clear description, linked issues, and completed checklist.

## Security & Configuration Tips

- Never commit secrets. Use environment variables or a local `.env` (gitignored) for provider API keys.
