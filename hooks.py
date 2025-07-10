import os
import re
from pathlib import Path


def get_nav_files(nav_config, docs_dir):
    """Extract file paths from mkdocs navigation config in order"""
    files = []

    def extract_files(nav_item):
        if isinstance(nav_item, dict):
            for key, value in nav_item.items():
                if isinstance(value, str):
                    # This is a file reference
                    if value.endswith(".md"):
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


def clean_markdown_content(content, file_path):
    """Clean markdown content for better concatenation"""
    # Remove mkdocs-specific directives
    content = re.sub(r"^\s*\[\[TOC\]\]\s*$", "", content, flags=re.MULTILINE)

    # Remove or replace relative links that won't work in concatenated format
    # Convert relative md links to section references where possible
    content = re.sub(r"\[([^\]]+)\]\(([^)]+\.md)\)", r"[\1](#\2)", content)

    # Add file path as a comment for reference
    content = f"<!-- Source: {file_path} -->\n\n{content}"

    return content


def get_file_description(file_path, docs_dir):
    """Extract description from markdown file's first paragraph or heading"""
    full_path = docs_dir / file_path

    if not full_path.exists():
        return ""

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for description in frontmatter or first paragraph
        lines = content.split("\n")
        description = ""

        # Skip frontmatter if present
        start_idx = 0
        if lines and lines[0].strip() == "---":
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == "---":
                    start_idx = i + 1
                    break

        # Find first meaningful paragraph after headings
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith("#") and not line.startswith("!"):
                # Clean up the line
                line = re.sub(
                    r"\[([^\]]+)\]\([^)]+\)", r"\1", line
                )  # Remove markdown links
                line = re.sub(r"`([^`]+)`", r"\1", line)  # Remove code formatting
                line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)  # Remove bold formatting
                line = re.sub(r"<[^>]+>", "", line)  # Remove HTML tags
                description = line.strip()
                if len(description) > 20:  # Only use if substantial
                    break

        # Fallback to title-based description
        if not description:
            path_obj = Path(file_path)
            name = path_obj.stem.replace("_", " ").replace("-", " ").title()
            if "index" in path_obj.name.lower():
                name = path_obj.parent.name.replace("_", " ").replace("-", " ").title()
            description = f"{name} documentation"

        return description[:100] + ("..." if len(description) > 100 else "")

    except Exception:
        return ""


def organize_files_by_section(nav_config, docs_dir):
    """Organize files into sections based on navigation structure"""
    sections = {
        "Getting Started": [],
        "Core Documentation": [],
        "Framework-Specific Guides": [],
        "Practical Examples": [],
        "API Reference": [],
        "Optional": [],
    }

    def categorize_file(file_path, nav_label=""):
        """Categorize a file based on its path and navigation context"""
        path_parts = Path(file_path).parts

        # Getting Started
        if (
            file_path == "index.md"
            or "quickstart" in file_path.lower()
            or "getting" in file_path.lower()
        ):
            return "Getting Started"

        # Framework-Specific Guides
        if "frameworks" in path_parts:
            return "Framework-Specific Guides"

        # Practical Examples
        if "cookbook" in path_parts or "examples" in path_parts:
            return "Practical Examples"

        # API Reference
        if "api" in path_parts:
            return "API Reference"

        # Core Documentation (agents, tools, callbacks, tracing, evaluation, serving)
        if any(
            core in path_parts
            for core in [
                "agents",
                "tools",
                "callbacks",
                "tracing",
                "evaluation",
                "serving",
            ]
        ):
            return "Core Documentation"

        # Optional for top-level auxiliary files
        if len(path_parts) == 1 and file_path != "index.md":
            return "Optional"

        return "Core Documentation"  # Default fallback

    def process_nav_item(item, parent_label=""):
        """Recursively process navigation items"""
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str) and value.endswith(".md"):
                    section = categorize_file(value, key)
                    sections[section].append(value)
                elif isinstance(value, list):
                    process_nav_item(value, key)
        elif isinstance(item, list):
            for sub_item in item:
                process_nav_item(sub_item, parent_label)

    # Process navigation items
    process_nav_item(nav_config)

    # Add any files not in navigation
    all_md_files = []
    for root, dirs, files in os.walk(docs_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for file in files:
            if file.endswith(".md"):
                rel_path = os.path.relpath(os.path.join(root, file), docs_dir)
                all_md_files.append(rel_path)

    # Add missing files to appropriate sections
    for file_path in all_md_files:
        found = False
        for section_files in sections.values():
            if file_path in section_files:
                found = True
                break

        if not found:
            section = categorize_file(file_path)
            sections[section].append(file_path)

    return sections


def generate_llms_txt(
    site_dir, docs_dir, nav_config, site_url="https://mozilla-ai.github.io/any-agent"
):
    """Generate the structured llms.txt file with curated links"""

    # Organize files by section
    sections = organize_files_by_section(nav_config, docs_dir)

    llms_content = [
        "# any-agent",
        "",
        "> A Python library providing a single interface to different agent frameworks. Supports multiple agent frameworks including OpenAI, LangChain, LlamaIndex, Google ADK, smolagents, and TinyAgent with unified APIs for creation, execution, and evaluation.",
        "",
        "any-agent simplifies working with different agent frameworks by providing a consistent interface across all supported platforms. The library includes built-in tools, comprehensive tracing capabilities, evaluation frameworks, and serving options for production deployment.",
        "",
    ]

    # Generate sections dynamically
    section_order = [
        "Getting Started",
        "Core Documentation",
        "Framework-Specific Guides",
        "Practical Examples",
        "API Reference",
        "Optional",
    ]

    for section_name in section_order:
        files = sections.get(section_name, [])
        if not files:
            continue

        llms_content.append(f"## {section_name}")
        llms_content.append("")

        for file_path in files:
            # Convert file path to markdown URL (following MCP approach)
            if file_path == "index.md":
                url = f"{site_url}/index.md"
            else:
                url = f"{site_url}/{file_path}"

            # Get title from file name or path
            if file_path == "index.md":
                title = "Documentation Home"
            else:
                title = Path(file_path).stem.replace("_", " ").replace("-", " ").title()
                if title.lower() == "index":
                    title = (
                        Path(file_path)
                        .parent.name.replace("_", " ")
                        .replace("-", " ")
                        .title()
                    )

            # Get description
            description = get_file_description(file_path, docs_dir)

            # Add to content
            llms_content.append(f"- [{title}]({url}): {description}")

        llms_content.append("")

    # Add external optional resources
    llms_content.extend(
        [
            "## Optional",
            "",
            "- [Contributing Guidelines](https://github.com/mozilla-ai/any-agent/blob/main/CONTRIBUTING.md): How to contribute to the project",
            "- [Code of Conduct](https://github.com/mozilla-ai/any-agent/blob/main/CODE_OF_CONDUCT.md): Community guidelines",
            "- [GitHub Repository](https://github.com/mozilla-ai/any-agent): Source code and issue tracking",
            "- [PyPI Package](https://pypi.org/project/any-agent/): Package installation and versions",
        ]
    )

    llms_txt_dest = site_dir / "llms.txt"
    try:
        with open(llms_txt_dest, "w", encoding="utf-8") as f:
            f.write("\n".join(llms_content))

    except Exception as e:
        pass


def generate_llms_full_txt(docs_dir, site_dir, nav_config):
    """Generate llms-full.txt by concatenating all markdown documentation"""

    # Extract files from navigation in order
    nav_files = get_nav_files(nav_config, docs_dir)

    # Also get any additional markdown files not in navigation
    all_md_files = []
    for root, dirs, files in os.walk(docs_dir):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for file in files:
            if file.endswith(".md"):
                rel_path = os.path.relpath(os.path.join(root, file), docs_dir)
                all_md_files.append(rel_path)

    # Combine nav files with any additional files, maintaining nav order
    ordered_files = []
    for file in nav_files:
        if file in all_md_files:
            ordered_files.append(file)

    # Add any remaining files not in navigation
    for file in all_md_files:
        if file not in ordered_files:
            ordered_files.append(file)

    # Generate the llms-full.txt content
    llms_full_content = []

    # Add header
    llms_full_content.append("# any-agent Documentation")
    llms_full_content.append("")
    llms_full_content.append(
        "> Complete documentation for any-agent - A Python library providing a single interface to different agent frameworks."
    )
    llms_full_content.append("")
    llms_full_content.append(
        "This file contains all documentation pages concatenated for easy consumption by AI systems."
    )
    llms_full_content.append("")
    llms_full_content.append("---")
    llms_full_content.append("")

    # Process each markdown file
    for file_path in ordered_files:
        full_path = docs_dir / file_path

        if full_path.exists():
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Clean and process content
                cleaned_content = clean_markdown_content(content, file_path)

                # Add section separator
                llms_full_content.append(f"## {file_path}")
                llms_full_content.append("")
                llms_full_content.append(cleaned_content)
                llms_full_content.append("")
                llms_full_content.append("---")
                llms_full_content.append("")

            except Exception as e:
                pass

    # Write the combined content to llms-full.txt
    llms_full_txt_dest = site_dir / "llms-full.txt"
    try:
        with open(llms_full_txt_dest, "w", encoding="utf-8") as f:
            f.write("\n".join(llms_full_content))

    except Exception as e:
        pass


def copy_markdown_files(docs_dir, site_dir):
    """Copy markdown files to site directory so they're accessible at .md URLs"""
    import shutil

    for root, dirs, files in os.walk(docs_dir):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for file in files:
            if file.endswith(".md"):
                # Get relative path from docs directory
                rel_path = os.path.relpath(os.path.join(root, file), docs_dir)

                # Create destination path in site directory
                dest_path = site_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the markdown file
                source_path = Path(root) / file
                try:
                    shutil.copy2(source_path, dest_path)
                except Exception as e:
                    pass


def on_post_build(config, **kwargs):
    """Generate both llms.txt and llms-full.txt files"""
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])

    # Get navigation configuration
    nav_config = config.get("nav", [])

    # Copy markdown files to site directory for direct access
    copy_markdown_files(docs_dir, site_dir)

    # Generate structured llms.txt
    generate_llms_txt(site_dir, docs_dir, nav_config)

    # Generate complete llms-full.txt
    generate_llms_full_txt(docs_dir, site_dir, nav_config)
