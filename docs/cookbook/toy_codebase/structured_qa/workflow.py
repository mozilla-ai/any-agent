from pathlib import Path

from loguru import logger
from rapidfuzz import process

from structured_qa.model_loaders import LlamaModel


def get_matching_section(response, section_names):
    """
    Use string similarity to find the most similar section_name.
    """
    return process.extractOne(response, section_names)[0]


def find_retrieve_answer(
    question: str,
    model: LlamaModel,
    sections_dir: str,
    find_prompt: str,
    answer_prompt: str,
    max_sections_to_check: int | None = None,
) -> tuple[str, list[str]] | tuple[None, list[str]]:
    """
    Workflow to find the relevant section, retrieve the information, and answer the question.

    Args:
        question (str): The question to answer.
        model (LlamaModel): The model to use for generating completions.
        sections_dir (str): The directory containing the sections.
            See [`document_to_sections_dir`][structured_qa.preprocessing.document_to_sections_dir].
            Structure of the sections directory:

            ```
            sections_dir/
                section_1.txt
                section_2.txt
                ...
            ```
        find_prompt (str): The prompt for finding the section.

            See [`FIND_PROMPT`][structured_qa.config.FIND_PROMPT].
        answer_prompt (str): The prompt for answering the question.

            See [`ANSWER_PROMPT`][structured_qa.config.ANSWER_PROMPT].
        max_sections_to_check (int, optional): The maximum number of sections to check before giving up.
            Defaults to None.
            If None, it will check  up to a maximum of 20 sections until it finds the answer.

    Returns:
        tuple[str, list[str]] | tuple[None, list[str]]:

            If the answer is found, the tuple contains the answer and the sections checked.
            If the answer is not found, the tuple contains None and the sections checked
    """
    sections_dir = Path(sections_dir)
    sections_names = [section.stem for section in sections_dir.glob("*.txt")]
    current_info = None
    current_section = None

    if max_sections_to_check is None:
        max_sections_to_check = min(20, len(sections_names))

    sections_checked = []
    while len(sections_checked) <= max_sections_to_check:
        logger.debug(f"Current information available: {current_info}")
        if not current_info:
            logger.debug("Finding section")
            finding_section = True
            question_part, *options = question.split("?")
            messages = [
                {
                    "role": "system",
                    "content": find_prompt.format(SECTIONS="\n".join(sections_names)),
                },
                {"role": "user", "content": question_part},
            ]
        else:
            logger.debug("Answering question")
            finding_section = False
            messages = [
                {
                    "role": "system",
                    "content": answer_prompt.format(CURRENT_INFO=current_info),
                },
                {"role": "user", "content": question},
            ]

        try:
            response = model.get_response(messages)
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            return "Generation Error", sections_checked

        if finding_section:
            response = response.strip()
            if not sections_names:
                return "NOT FOUND", sections_checked
            section_name = get_matching_section(response, sections_names)
            logger.debug(f"Retrieving section: {section_name}")
            section_content = (sections_dir / f"{section_name}.txt").read_text()
            current_section = section_name
            current_info = section_content
            sections_checked.append(section_name)

        else:
            if "MORE INFO" in response.upper():
                current_info = None
                sections_names.remove(current_section)
                continue
            else:
                return response, sections_checked

    return "NOT FOUND", sections_checked
