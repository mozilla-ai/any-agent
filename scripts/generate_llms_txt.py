#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt files after the Starlight docs build.

These files follow the llmstxt.org standard for making documentation
accessible to AI systems.

Usage:
    python scripts/generate_llms_txt.py
"""

import os
import re
import sys
from pathlib import Path

DOCS_CONTENT_DIR = Path(__file__).parent.parent / "docs"
BUILD_OUTPUT_DIR = Path(__file__).parent.parent / "site"
BASE_URL = (
    "https://raw.githubusercontent.com/mozilla-ai/any-agent/refs/heads/main/docs/"
)
MARKDOWN_EXTENSION = ".md"
ENCODING = "utf-8"
TOC_PATTERN = r"^\s*\[\[TOC\]\]\s*$"

ORDERED_FILES = [
    "index.md",
    "agents/index.md",
    "agents/models.md",
    "agents/callbacks.md",
    "agents/tools.md",
    "agents/frameworks/agno.md",
    "agents/frameworks/google-adk.md",
    "agents/frameworks/langchain.md",
    "agents/frameworks/llama-index.md",
    "agents/frameworks/openai.md",
    "agents/frameworks/smolagents.md",
    "agents/frameworks/tinyagent.md",
    "frameworks.md",
    "tracing.md",
    "evaluation.md",
    "serving.md",
    "cookbook/your-first-agent.md",
    "cookbook/your-first-agent-evaluation.md",
    "cookbook/callbacks.md",
    "cookbook/mcp-agent.md",
    "cookbook/serve-a2a.md",
    "cookbook/a2a-as-tool.md",
    "cookbook/agent-with-local-llm.md",
]


TITLE_OVERRIDES = {
    "agents/frameworks/google-adk.md": "Agents - Frameworks - Google ADK",
    "agents/frameworks/llama-index.md": "Agents - Frameworks - LlamaIndex",
    "agents/frameworks/openai.md": "Agents - Frameworks - OpenAI",
    "agents/frameworks/smolagents.md": "Agents - Frameworks - smolagents",
    "agents/frameworks/tinyagent.md": "Agents - Frameworks - TinyAgent",
    "cookbook/mcp-agent.md": "Cookbook - MCP Agent",
    "cookbook/a2a-as-tool.md": "Cookbook - A2A as Tool",
    "cookbook/agent-with-local-llm.md": "Cookbook - Local LLM Agent",
}


def create_file_title(file_path: str) -> str:
    """Create a clean title from file path."""
    if file_path == "index.md":
        return "Introduction"
    if file_path in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[file_path]

    name = file_path.replace(MARKDOWN_EXTENSION, "")
    return name.replace("_", " ").replace("-", " ").replace("/", " - ").title()


def extract_description_from_markdown(content: str) -> str:
    """Extract a description from markdown content.

    Prefers the frontmatter 'description' field. Falls back to the first
    prose paragraph after the H1, skipping fenced code blocks and other
    non-prose lines.
    """
    if not content:
        return ""

    # Prefer frontmatter description
    fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if fm_match:
        for fm_line in fm_match.group(1).splitlines():
            m = re.match(r"^description\s*:\s*(.+)", fm_line)
            if m:
                return m.group(1).strip().strip("\"'")

    # Extract H1 title as fallback if no prose description found below
    h1_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    h1_title = h1_match.group(1).strip() if h1_match else ""

    lines = content.split("\n")
    title_found = False
    in_frontmatter = False
    in_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue

        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if not stripped:
            continue

        if stripped.startswith("# ") and not title_found:
            title_found = True
            continue

        if not title_found:
            continue

        if (
            stripped.startswith(
                (
                    "!!! ",
                    "<",
                    ":::",
                    "##",
                    "{%",
                    "---",
                    "|",
                    "- ",
                    "* ",
                    "import ",
                    "http://",
                    "https://",
                    "![",
                    "_(",
                )
            )
            or re.match(r"^\[!\[", stripped)
            or re.match(r"^\[https?://", stripped)
            or (stripped.startswith("[") and stripped.endswith("]"))
            or re.match(r"^\d+\.", stripped)
        ):
            continue

        if len(stripped) > 20:
            description = stripped
            description = re.sub(r"\*\*([^*]+)\*\*", r"\1", description)
            description = re.sub(r"\*([^*]+)\*", r"\1", description)
            description = re.sub(r"`([^`]+)`", r"\1", description)
            return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", description)

    return h1_title


def clean_markdown_content(content: str, file_path: str) -> str:
    """Clean markdown content for concatenation."""
    # Remove frontmatter
    if content.startswith("---"):
        end_idx = content.find("---", 3)
        if end_idx != -1:
            content = content[end_idx + 3 :].lstrip()

    # Remove Astro/MDX imports
    content = re.sub(r"^import\s+.*$", "", content, flags=re.MULTILINE)

    # Remove GitBook block tags: {% hint %}, {% content-ref %}, {% tab %}, etc.
    content = re.sub(r"\{%[^%]*%\}", "", content)

    # Remove JSX/MDX components: <CardGrid>, <Card ...>, </Card>, etc.
    content = re.sub(r"</?[A-Z][A-Za-z]*[^>]*>", "", content)

    # Remove Starlight admonitions: :::note, :::tip, :::caution, :::
    content = re.sub(r"^:::[a-z]*.*$", "", content, flags=re.MULTILINE)

    content = re.sub(TOC_PATTERN, "", content, flags=re.MULTILINE)

    # Collapse runs of blank lines left by removals
    content = re.sub(r"\n{3,}", "\n\n", content)

    return f"<!-- Source: {file_path} -->\n\n{content.strip()}"


def get_all_content_files() -> list[str]:
    """Get ordered list of content files, including any not in the explicit list."""
    ordered = list(ORDERED_FILES)

    for root, _, files in os.walk(DOCS_CONTENT_DIR):
        for f in files:
            if f.endswith(MARKDOWN_EXTENSION):
                rel_path = os.path.relpath(os.path.join(root, f), DOCS_CONTENT_DIR)
                if rel_path not in ordered:
                    ordered.append(rel_path)

    return ordered


def generate_llms_txt(ordered_files: list[str]) -> str:
    """Generate llms.txt content."""
    lines = ["# any-agent", "", "## Docs", ""]

    for file_path in ordered_files:
        full_path = DOCS_CONTENT_DIR / file_path
        if not full_path.exists():
            continue

        txt_url = f"{BASE_URL}{file_path}"
        title = create_file_title(file_path)
        content = full_path.read_text(encoding=ENCODING)
        description = extract_description_from_markdown(content)

        if description:
            lines.append(f"- [{title}]({txt_url}): {description}")
        else:
            lines.append(f"- [{title}]({txt_url})")

    return "\n".join(lines)


def generate_llms_full_txt(ordered_files: list[str]) -> str:
    """Generate llms-full.txt by concatenating all documentation."""
    sections = [
        "# any-agent Documentation",
        "",
        "> Complete documentation for any-agent - A Python library providing a single interface to different agent frameworks.",
        "",
        "This file contains all documentation pages concatenated for easy consumption by AI systems.",
        "",
        "---",
        "",
    ]

    for file_path in ordered_files:
        full_path = DOCS_CONTENT_DIR / file_path
        if not full_path.exists():
            continue

        content = full_path.read_text(encoding=ENCODING)
        cleaned = clean_markdown_content(content, file_path)
        title = create_file_title(file_path)
        sections.extend([f"## {title}", "", cleaned, "", "---", ""])

    return "\n".join(sections)


def main() -> int:
    """Generate llms.txt and llms-full.txt files."""
    ordered_files = get_all_content_files()

    BUILD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llms_txt = generate_llms_txt(ordered_files)
    llms_txt_path = BUILD_OUTPUT_DIR / "llms.txt"
    llms_txt_path.write_text(llms_txt, encoding=ENCODING)
    print(f"Generated {llms_txt_path}")

    llms_full_txt = generate_llms_full_txt(ordered_files)
    llms_full_path = BUILD_OUTPUT_DIR / "llms-full.txt"
    llms_full_path.write_text(llms_full_txt, encoding=ENCODING)
    print(f"Generated {llms_full_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
