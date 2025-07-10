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


def on_post_build(config, **kwargs):
    """Generate llms-full.txt files"""
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])

    # Get navigation configuration
    nav_config = config.get("nav", [])

    # Generate complete llms-full.txt
    generate_llms_full_txt(docs_dir, site_dir, nav_config)
