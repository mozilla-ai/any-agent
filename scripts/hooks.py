"""Generate llms.txt and llms-full.txt files for the documentation.

It also serves markdown files as text files for LLM consumption.
It was initially written in collaboration with Claude 4 Sonnet.
"""

import re
from pathlib import Path

MARKDOWN_EXTENSION = ".md"
BASE_URL = (
    "https://raw.githubusercontent.com/mozilla-ai/any-agent/refs/heads/main/docs/"
)
EXCLUDED_DIRS = {".", "__pycache__"}
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


def get_ordered_files(nav_config, docs_dir: Path):
    """Get ordered list of markdown files based on navigation and additional files."""
    nav_files = get_nav_files(nav_config)
    all_md_files = [str(f.relative_to(docs_dir)) for f in docs_dir.rglob("*.md")]
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
    # Remove or replace relative links that won't work in concatenated format
    # Convert relative md links to section references where possible
    content = re.sub(MARKDOWN_LINK_PATTERN, MARKDOWN_LINK_REPLACEMENT, content)

    return f"<!-- Source: {file_path} -->\n\n{content}"


def get_file_description(file_path, docs_dir):
    """Get descriptive text extracted from the actual file."""
    full_path = docs_dir / file_path

    if not full_path.exists():
        return ""

    content = full_path.read_text()
    if content is None:
        return ""

    # Assume subheadings are a good enough description
    # or at least better than the first characters in the file.
    return " - ".join(
        line.replace("## ", "").strip()
        for line in content.split("\n")
        if line.startswith("## ")
    )


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
    llms_txt_dest.write_text("\n".join(llms_txt_content))


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
            content = full_path.read_text()
            if content is not None:
                cleaned_content = clean_markdown_content(content, file_path)

                # Add section separator
                llms_full_content.extend(
                    [f"## {file_path}", "", cleaned_content, "", "---", ""]
                )

    llms_full_txt_dest = site_dir / "llms-full.txt"
    llms_full_txt_dest.write_text("\n".join(llms_full_content))


def on_post_build(config, **kwargs):
    """Generate llms.txt and llms-full.txt files, and serve markdown as text."""
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])

    nav_config = config.get("nav", [])

    generate_llms_txt(docs_dir, site_dir, nav_config)

    generate_llms_full_txt(docs_dir, site_dir, nav_config)
