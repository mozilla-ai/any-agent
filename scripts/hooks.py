"""Generate llms.txt and llms-full.txt files for the documentation.

It also serves markdown files as text files for LLM consumption.
It was initially written in collaboration with Claude 4 Sonnet.
"""

import os
import re
from pathlib import Path

MARKDOWN_EXTENSION = ".md"
BASE_URL = (
    "https://raw.githubusercontent.com/mozilla-ai/any-agent/refs/heads/main/docs/"
)
ENCODING = "utf-8"
EXCLUDED_DIRS = {".", "__pycache__"}
TOC_PATTERN = r"^\s*\[\[TOC\]\]\s*$"
MARKDOWN_LINK_PATTERN = r"\[([^\]]+)\]\(([^)]+\.md)\)"
MARKDOWN_LINK_REPLACEMENT = r"[\1](#\2)"


def get_nav_files(nav_config):
    """Extract file paths from mkdocs navigation config in order."""
    files = []

    def extract_files(nav_item):
        if isinstance(nav_item, dict):
            for _, value in nav_item.items():
                if isinstance(value, str):
                    # This is a file reference
                    if value.endswith(MARKDOWN_EXTENSION):
                        files.append(value)
                elif isinstance(value, list):
                    # This is a nested section
                    for item in value:
                        extract_files(item)
        elif isinstance(nav_item, list):
            for item in nav_item:
                extract_files(item)

    extract_files(nav_config)
    return files


def get_all_markdown_files(docs_dir):
    """Get all markdown files in the documentation directory."""
    all_md_files = []
    for root, dirs, files in os.walk(docs_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for file in files:
            if file.endswith(MARKDOWN_EXTENSION):
                rel_path = os.path.relpath(os.path.join(root, file), docs_dir)
                all_md_files.append(rel_path)
    return all_md_files


def get_ordered_files(nav_config, docs_dir):
    """Get ordered list of markdown files based on navigation and additional files."""
    nav_files = get_nav_files(nav_config)
    all_md_files = get_all_markdown_files(docs_dir)

    ordered_files = []
    for file in nav_files:
        if file in all_md_files:
            ordered_files.append(file)

    for file in all_md_files:
        if file not in ordered_files:
            ordered_files.append(file)

    return ordered_files


def clean_markdown_content(content, file_path):
    """Clean markdown content for better concatenation."""
    # Remove mkdocs-specific directives
    content = re.sub(TOC_PATTERN, "", content, flags=re.MULTILINE)

    # Remove or replace relative links that won't work in concatenated format
    # Convert relative md links to section references where possible
    content = re.sub(MARKDOWN_LINK_PATTERN, MARKDOWN_LINK_REPLACEMENT, content)

    return f"<!-- Source: {file_path} -->\n\n{content}"


def extract_description_from_markdown(content):
    """Extract a description from markdown content."""
    if not content:
        return ""

    lines = content.split("\n")
    title_found = False
    description_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("# ") and not title_found:
            title_found = True
            continue

        if not title_found:
            continue

        if (
            stripped.startswith("!!! ")  # Admonitions  # noqa: PIE810
            or stripped.startswith("<")  # HTML tags
            or stripped.startswith(":::")  # API references
            or stripped.startswith("##")  # Subheadings
            or stripped.startswith("```")  # Code blocks
            or stripped.startswith("---")  # Horizontal rules
            or stripped.startswith("|")  # Tables
            or stripped.startswith("- ")  # Lists
            or stripped.startswith("* ")  # Lists
            or (stripped.startswith("[") and stripped.endswith("]"))  # Standalone links
            or re.match(r"^\d+\.", stripped)  # Numbered lists
        ):
            continue

        if len(stripped) > 20:  # Minimum length for a meaningful description
            description_lines.append(stripped)
            # For now, just take the first good paragraph
            break

    if description_lines:
        description = " ".join(description_lines)
        description = re.sub(r"\*\*([^*]+)\*\*", r"\1", description)  # Bold
        description = re.sub(r"\*([^*]+)\*", r"\1", description)  # Italic
        description = re.sub(r"`([^`]+)`", r"\1", description)  # Code
        return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", description)  # Links

    return ""


def get_file_description(file_path, docs_dir):
    """Get descriptive text extracted from the actual file."""
    full_path = docs_dir / file_path

    if not full_path.exists():
        return ""

    content = read_file_content(full_path)
    if content is None:
        return ""

    return extract_description_from_markdown(content)


def read_file_content(file_path):
    """Safely read file content with error handling."""
    with open(file_path, encoding=ENCODING) as f:
        return f.read()


def write_file_content(file_path, content):
    """Safely write file content with error handling."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding=ENCODING) as f:
        f.write(content)


def create_file_title(file_path):
    """Create a clean title from file path."""
    if file_path == "index.md":
        return "Introduction"

    return (
        file_path.replace(MARKDOWN_EXTENSION, "")
        .replace("_", " ")
        .replace("/", " - ")
        .title()
    )


def generate_llms_txt(docs_dir, site_dir, nav_config):
    """Generate llms.txt file following llmstxt.org standards."""
    ordered_files = get_ordered_files(nav_config, docs_dir)

    llms_txt_content = []

    llms_txt_content.append("# any-agent")
    llms_txt_content.append("")
    llms_txt_content.append("## Docs")
    llms_txt_content.append("")

    for file_path in ordered_files:
        txt_url = f"{BASE_URL}{file_path}"

        title = create_file_title(file_path)
        description = get_file_description(file_path, docs_dir)

        if description:
            llms_txt_content.append(f"- [{title}]({txt_url}): {description}")
        else:
            llms_txt_content.append(f"- [{title}]({txt_url})")

    llms_txt_dest = site_dir / "llms.txt"
    write_file_content(llms_txt_dest, "\n".join(llms_txt_content))


def generate_llms_full_txt(docs_dir, site_dir, nav_config):
    """Generate llms-full.txt by concatenating all markdown documentation."""
    ordered_files = get_ordered_files(nav_config, docs_dir)

    llms_full_content = []

    llms_full_content.extend(
        [
            "# any-agent Documentation",
            "",
            "> Complete documentation for any-agent - A Python library providing a single interface to different agent frameworks.",
            "",
            "This file contains all documentation pages concatenated for easy consumption by AI systems.",
            "",
            "---",
            "",
        ]
    )

    for file_path in ordered_files:
        full_path = docs_dir / file_path

        if full_path.exists():
            content = read_file_content(full_path)
            if content is not None:
                cleaned_content = clean_markdown_content(content, file_path)

                # Add section separator
                llms_full_content.extend(
                    [f"## {file_path}", "", cleaned_content, "", "---", ""]
                )

    llms_full_txt_dest = site_dir / "llms-full.txt"
    write_file_content(llms_full_txt_dest, "\n".join(llms_full_content))


def on_post_build(config, **kwargs):
    """Generate llms.txt and llms-full.txt files, and serve markdown as text."""
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])

    nav_config = config.get("nav", [])

    generate_llms_txt(docs_dir, site_dir, nav_config)

    generate_llms_full_txt(docs_dir, site_dir, nav_config)
