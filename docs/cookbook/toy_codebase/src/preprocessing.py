import re
from collections import defaultdict
from pathlib import Path

import pymupdf4llm

from loguru import logger


def split_markdown_by_headings(
    markdown_text, heading_patterns: list[str] | None = None
) -> dict[str, str]:
    """Splits a markdown document into sections based on specified heading patterns.

    Args:
        markdown_text (str): The markdown document as a single string.
        heading_patterns (str, optional): A list of regex patterns representing heading markers
            in the markdown document.
            Defaults to None.
            If None, the default patterns are used.

    Returns:
        dict[str, str]: A dictionary where the keys are the section names and the values are the section contents.
    """
    if heading_patterns is None:
        heading_patterns = [
            r"^#\s+(.+)$",
            r"^##\s+(.+)$",
            r"^###\s+(.+)$",
            r"^####\s+(.+)$",
            r"^\*\*[\d\.]+\.\*\*\s*\*\*(.+)\*\*$",
        ]

    sections = defaultdict(str)

    heading_text = "INTRO"
    for line in markdown_text.splitlines():
        line = line.strip()
        if not line:
            continue
        for pattern in heading_patterns:
            match = re.match(pattern, line)
            if match:
                heading_text = match.group(1)[:100]
                break
        sections[heading_text] += f"{line}\n"

    return sections


@logger.catch(reraise=True)
def document_to_sections_dir(input_file: str, output_dir: str) -> list[str]:
    """
    Convert a document to a directory of sections.

    Uses [pymupdf4llm](https://pypi.org/project/pymupdf4llm/) to convert input_file to markdown.
    Then uses [`split_markdown_by_headings`][structured_qa.preprocessing.split_markdown_by_headings] to split the markdown into sections based on the headers.

    Args:
        input_file: Path to the input document.
        output_dir: Path to the output directory.
            Structure of the output directory:

            ```
            output_dir/
                section_1.txt
                section_2.txt
                ...
            ```

    Returns:
        List of section names.
    """

    logger.info(f"Converting {input_file}")
    md_text = pymupdf4llm.to_markdown(input_file)
    Path("debug.md").write_text(md_text)
    logger.success("Converted")

    logger.info("Extracting sections")
    sections = split_markdown_by_headings(
        md_text,
    )
    logger.success(f"Found {len(sections)} sections")
    logger.info(f"Writing sections to {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for section_name, section_content in sections.items():
        (output_dir / f"{section_name.replace('/', '_')}.txt").write_text(
            section_content
        )
    logger.success("Done")

    return sections.keys()
