"""Convert Astro/Starlight docs to plain Markdown for GitBook.

Reads from docs/src/content/docs/, strips MDX/JSX syntax, and writes
to a site/ output directory along with a SUMMARY.md for GitBook navigation.

Usage:
    python scripts/convert_to_gitbook.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

DOCS_SRC = Path("docs/src/content/docs")
SITE_DIR = Path("site")

# Mirrors the sidebar in astro.config.mjs — used to build SUMMARY.md
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


ADMONITION_STYLES = {
    "note": "info",
    "tip": "success",
    "caution": "warning",
    "danger": "danger",
}


def convert_content(content: str) -> str:
    """Strip Astro/MDX-specific syntax from a file's content."""
    lines = content.split("\n")
    out = []

    for line in lines:
        # Strip MDX import statements
        if re.match(r"^import\s+.*from\s+['\"].*['\"];?\s*$", line):
            continue

        # Convert Starlight admonitions to GitBook hints
        # Matches :::note, :::tip, :::tip[Title], :::caution, :::danger
        admonition_open = re.match(r"^:::(\w+)(?:\[([^\]]*)\])?$", line.strip())
        if admonition_open:
            kind = admonition_open.group(1).lower()
            style = ADMONITION_STYLES.get(kind, "info")
            out.append(f'{{% hint style="{style}" %}}')
            continue

        # Closing :::
        if line.strip() == ":::":
            out.append("{% endhint %}")
            continue

        # Strip Tabs wrapper (keep contents)
        if line.strip() in ("<Tabs>", "</Tabs>"):
            continue

        # Convert TabItem to a bold header, drop closing tag
        tab_match = re.match(r"^\s*<TabItem\s+label=[\"']([^\"']+)[\"']>", line)
        if tab_match:
            out.append(f"\n**{tab_match.group(1)}**\n")
            continue
        if line.strip() == "</TabItem>":
            continue

        # Strip iframe tags (not renderable in GitBook)
        if re.match(r"^\s*<iframe\s+", line):
            out.append("*[Interactive trace — view on the docs site]*")
            continue

        # Strip other JSX self-closing or block tags (e.g. <Badge />, <CardGrid>)
        if re.match(r"^\s*<[A-Z][^>]*/>\s*$", line) or re.match(
            r"^\s*</?[A-Z][^>]*>\s*$", line
        ):
            continue

        out.append(line)

    return "\n".join(out)


def process_file(src: Path, dst: Path) -> None:
    """Convert a single .md or .mdx file and write to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    content = src.read_text()
    content = convert_content(content)
    # Always write as .md regardless of source extension
    dst.with_suffix(".md").write_text(content)


def main() -> None:
    """Convert all docs and write to site/."""
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()

    # Process all .md and .mdx source files
    for src in sorted(DOCS_SRC.rglob("*")):
        if src.suffix not in (".md", ".mdx"):
            continue
        rel = src.relative_to(DOCS_SRC)
        dst = SITE_DIR / rel.with_suffix(".md")
        print(f"  {rel}")
        process_file(src, dst)

    # Write SUMMARY.md and GitBook config
    (SITE_DIR / "SUMMARY.md").write_text(SUMMARY)
    shutil.copy(".gitbook.yaml", SITE_DIR / ".gitbook.yaml")

    print(f"\nDone — {len(list(SITE_DIR.rglob('*.md')))} files written to {SITE_DIR}/")


if __name__ == "__main__":
    main()
