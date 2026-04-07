"""Build the GitBook site output from docs/src/content/docs/.

Copies the docs source into site/, copies static assets, and writes
SUMMARY.md for GitBook navigation.

Usage:
    python scripts/convert_to_gitbook.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

DOCS_SRC = Path("docs/src/content/docs")
PUBLIC_SRC = Path("docs/public")
SITE_DIR = Path("site")

SUMMARY = """\
# Table of Contents

* [Introduction](index.md)

## Agents

* [Define and Run Agents](agents/index.md)
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
    """Copy all Markdown files from docs/src/content/docs/ into site/."""
    for src in sorted(DOCS_SRC.rglob("*.md")):
        rel = src.relative_to(DOCS_SRC)
        dst = SITE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {rel}")


def generate_frameworks_index() -> None:
    """Generate the frameworks index page."""
    dst = SITE_DIR / "agents" / "frameworks" / "index.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(FRAMEWORKS_INDEX)
    print("  agents/frameworks/index.md (generated)")


def copy_assets() -> None:
    """Copy static assets from docs/public into site/."""
    for subdir in ("images", "traces"):
        src = PUBLIC_SRC / subdir
        if src.exists():
            shutil.copytree(src, SITE_DIR / subdir)
            print(f"  Copied {len(list(src.rglob('*')))} assets from public/{subdir}/")


def main() -> None:
    """Build site/ from docs source and static assets."""
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()

    copy_docs()
    generate_frameworks_index()
    copy_assets()

    (SITE_DIR / "SUMMARY.md").write_text(SUMMARY)
    shutil.copy(".gitbook.yaml", SITE_DIR / ".gitbook.yaml")

    print(f"\nDone — {len(list(SITE_DIR.rglob('*.md')))} files written to {SITE_DIR}/")


if __name__ == "__main__":
    main()
