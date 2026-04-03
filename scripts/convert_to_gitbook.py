"""Build the GitBook site output from docs-md/.

Copies docs-md/ (pre-converted GitBook Markdown) into site/, generates the
API reference from Python docstrings, copies static assets, and writes
SUMMARY.md for GitBook navigation.

When docs-md/ is eventually merged back into docs/, update DOCS_SRC to point
at the new location.

Usage:
    python scripts/convert_to_gitbook.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

DOCS_SRC = Path("docs-md")
API_SRC = Path("docs/src/content/docs/api")
PUBLIC_SRC = Path("docs/public")
SITE_DIR = Path("site")

SUMMARY = """\
# Table of Contents

* [Introduction](index.md)

## Agents

* [Defining and Running Agents](agents/index.md)
* [Models](agents/models.md)
* [Callbacks](agents/callbacks.md)
* [Frameworks](agents/frameworks/index.md)
  * [Agno](agents/frameworks/agno.md)
  * [Google ADK](agents/frameworks/google-adk.md)
  * [LangChain](agents/frameworks/langchain.md)
  * [LlamaIndex](agents/frameworks/llama-index.md)
  * [OpenAI Agents SDK](agents/frameworks/openai.md)
  * [smolagents](agents/frameworks/smolagents.md)
  * [TinyAgent](agents/frameworks/tinyagent.md)
* [Tools](agents/tools.md)

## Core Concepts

* [Frameworks](frameworks.md)
* [Tracing](tracing.md)
* [Evaluation](evaluation.md)
* [Serving](serving.md)

## Cookbook

* [Your First Agent](cookbook/your-first-agent.md)
* [Your First Agent Evaluation](cookbook/your-first-agent-evaluation.md)
* [Using Callbacks](cookbook/callbacks.md)
* [MCP Agent](cookbook/mcp-agent.md)
* [Serve with A2A](cookbook/serve-a2a.md)
* [Use an Agent as a Tool (A2A)](cookbook/a2a-as-tool.md)
* [Local Agent](cookbook/agent-with-local-llm.md)

## API Reference

* [Agent](api/agent.md)
* [Callbacks](api/callbacks.md)
* [Config](api/config.md)
* [Evaluation](api/evaluation.md)
* [Logging](api/logging.md)
* [Serving](api/serving.md)
* [Tools](api/tools.md)
* [Tracing](api/tracing.md)
"""

FRAMEWORKS_INDEX = """\
# Agent Frameworks

any-agent supports multiple agent frameworks through a unified interface.

| Framework | Page |
|-----------|------|
| Agno | [agno.md](agno.md) |
| Google ADK | [google-adk.md](google-adk.md) |
| LangChain | [langchain.md](langchain.md) |
| LlamaIndex | [llama-index.md](llama-index.md) |
| OpenAI Agents SDK | [openai.md](openai.md) |
| smolagents | [smolagents.md](smolagents.md) |
| TinyAgent | [tinyagent.md](tinyagent.md) |
"""


def copy_docs() -> None:
    """Copy all Markdown files from docs-md/ into site/."""
    for src in sorted(DOCS_SRC.rglob("*.md")):
        rel = src.relative_to(DOCS_SRC)
        dst = SITE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {rel}")


def generate_frameworks_index() -> None:
    """Generate the frameworks index page (no source equivalent in docs-md)."""
    dst = SITE_DIR / "agents" / "frameworks" / "index.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(FRAMEWORKS_INDEX)
    print("  agents/frameworks/index.md (generated)")


def copy_api_docs() -> None:
    """Copy generated API docs from docs/src/content/docs/api/ into site/api/."""
    if not API_SRC.exists():
        print("  WARNING: API docs not found — run generate_api_docs.py first")
        return
    dst = SITE_DIR / "api"
    shutil.copytree(API_SRC, dst)
    print(f"  Copied {len(list(dst.rglob('*.md')))} API docs from {API_SRC}/")


def copy_assets() -> None:
    """Copy static assets from docs/public into site/."""
    for subdir in ("images", "traces"):
        src = PUBLIC_SRC / subdir
        if src.exists():
            shutil.copytree(src, SITE_DIR / subdir)
            print(f"  Copied {len(list(src.rglob('*')))} assets from public/{subdir}/")


def main() -> None:
    """Build site/ from docs-md/ and generated API docs."""
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()

    copy_docs()
    generate_frameworks_index()
    copy_api_docs()
    copy_assets()

    (SITE_DIR / "SUMMARY.md").write_text(SUMMARY)
    shutil.copy(".gitbook.yaml", SITE_DIR / ".gitbook.yaml")

    print(f"\nDone — {len(list(SITE_DIR.rglob('*.md')))} files written to {SITE_DIR}/")


if __name__ == "__main__":
    main()
