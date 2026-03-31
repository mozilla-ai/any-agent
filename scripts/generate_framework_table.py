#!/usr/bin/env python3
"""Generate the framework comparison table for the frameworks page.

Reads framework capabilities and injects a markdown table between marker
comments in the frameworks comparison page.

Usage:
    python scripts/generate_framework_table.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FRAMEWORKS_PAGE = REPO_ROOT / "docs" / "src" / "content" / "docs" / "frameworks.md"

START_MARKER = "<!-- FRAMEWORK-TABLE-START -->"
END_MARKER = "<!-- FRAMEWORK-TABLE-END -->"


# Framework data: (name, link, docs_url, features_dict)
FRAMEWORKS = [
    {
        "name": "Agno",
        "link": "/any-agent/agents/frameworks/agno/",
        "docs": "https://docs.agno.com/",
        "callable_tools": True,
        "mcp_tools": True,
        "composio_tools": True,
        "structured_output": True,
        "streaming": False,
        "multi_agent": False,
        "callbacks": True,
        "any_llm": True,
    },
    {
        "name": "Google ADK",
        "link": "/any-agent/agents/frameworks/google-adk/",
        "docs": "https://google.github.io/adk-docs/",
        "callable_tools": True,
        "mcp_tools": True,
        "composio_tools": True,
        "structured_output": True,
        "streaming": False,
        "multi_agent": False,
        "callbacks": True,
        "any_llm": True,
    },
    {
        "name": "LangChain",
        "link": "/any-agent/agents/frameworks/langchain/",
        "docs": "https://python.langchain.com/",
        "callable_tools": True,
        "mcp_tools": True,
        "composio_tools": True,
        "structured_output": True,
        "streaming": False,
        "multi_agent": False,
        "callbacks": True,
        "any_llm": True,
    },
    {
        "name": "LlamaIndex",
        "link": "/any-agent/agents/frameworks/llama-index/",
        "docs": "https://docs.llamaindex.ai/",
        "callable_tools": True,
        "mcp_tools": True,
        "composio_tools": True,
        "structured_output": True,
        "streaming": False,
        "multi_agent": False,
        "callbacks": True,
        "any_llm": True,
    },
    {
        "name": "OpenAI Agents SDK",
        "link": "/any-agent/agents/frameworks/openai/",
        "docs": "https://openai.github.io/openai-agents-python/",
        "callable_tools": True,
        "mcp_tools": True,
        "composio_tools": True,
        "structured_output": True,
        "streaming": False,
        "multi_agent": True,
        "callbacks": True,
        "any_llm": True,
    },
    {
        "name": "smolagents",
        "link": "/any-agent/agents/frameworks/smolagents/",
        "docs": "https://huggingface.co/docs/smolagents/",
        "callable_tools": True,
        "mcp_tools": True,
        "composio_tools": True,
        "structured_output": True,
        "streaming": False,
        "multi_agent": False,
        "callbacks": True,
        "any_llm": True,
    },
    {
        "name": "TinyAgent",
        "link": "/any-agent/agents/frameworks/tinyagent/",
        "docs": "https://github.com/mozilla-ai/any-agent/blob/main/src/any_agent/frameworks/tinyagent.py",
        "callable_tools": True,
        "mcp_tools": True,
        "composio_tools": True,
        "structured_output": True,
        "streaming": True,
        "multi_agent": False,
        "callbacks": True,
        "any_llm": True,
    },
]


def _check(val: bool) -> str:
    return "\u2705" if val else "\u274c"


def generate_table() -> str:
    """Generate the markdown table."""
    header = "| Framework | Docs | Callable Tools | MCP Tools | Composio Tools | Structured Output | Streaming | Multi-Agent (Handoffs) | Callbacks | any-llm Integration |"
    separator = "|-----------|------|----------------|-----------|----------------|-------------------|-----------|------------------------|-----------|---------------------|"

    rows = []
    for fw in FRAMEWORKS:
        row = (
            f"| [{fw['name']}]({fw['link']}) "
            f"| [docs]({fw['docs']}) "
            f"| {_check(fw['callable_tools'])} "
            f"| {_check(fw['mcp_tools'])} "
            f"| {_check(fw['composio_tools'])} "
            f"| {_check(fw['structured_output'])} "
            f"| {_check(fw['streaming'])} "
            f"| {_check(fw['multi_agent'])} "
            f"| {_check(fw['callbacks'])} "
            f"| {_check(fw['any_llm'])} |"
        )
        rows.append(row)

    return "\n".join([header, separator, *rows])


def main() -> int:
    """Inject generated table into frameworks page."""
    if not FRAMEWORKS_PAGE.exists():
        print(f"Frameworks page not found: {FRAMEWORKS_PAGE}", file=sys.stderr)
        return 1

    content = FRAMEWORKS_PAGE.read_text(encoding="utf-8")

    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)

    if start_idx == -1 or end_idx == -1:
        print("Marker comments not found in frameworks page", file=sys.stderr)
        return 1

    table = generate_table()
    new_content = (
        content[: start_idx + len(START_MARKER)]
        + "\n"
        + table
        + "\n"
        + content[end_idx:]
    )

    FRAMEWORKS_PAGE.write_text(new_content, encoding="utf-8")
    print(f"Updated {FRAMEWORKS_PAGE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
