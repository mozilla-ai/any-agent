from pathlib import Path

import yaml
from fire import Fire
from loguru import logger
from structured_qa.config import ANSWER_PROMPT, FIND_PROMPT, Config
from structured_qa.model_loaders import load_llama_cpp_model
from structured_qa.preprocessing import document_to_sections_dir
from structured_qa.workflow import find_retrieve_answer


@logger.catch(reraise=True)
def structured_qa(
    question: str | None = None,
    input_file: str | None = None,
    output_dir: str | None = None,
    model: str
    | None = "bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q8_0.gguf",
    find_prompt: str = FIND_PROMPT,
    answer_prompt: str = ANSWER_PROMPT,
    from_config: str | None = None,
):
    """
    Structured Question Answering.

    Split the input document into sections and answer the question based on the sections.

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
        model: Model identifier formatted as `owner/repo/file`.
            Must be hosted at the HuggingFace Hub in GGUF format.
            Defaults to bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q8_0.gguf.
        question: The question to answer.
        find_prompt: The prompt for finding the section.

            See [`FIND_PROMPT`][structured_qa.config.FIND_PROMPT].
        answer_prompt: The prompt for answering the question.

            See [`ANSWER_PROMPT`][structured_qa.config.ANSWER_PROMPT].
        from_config: The path to the config file.

            If provided, all other arguments will be ignored.
    """
    if from_config:
        raw_config = yaml.safe_load(Path(from_config).read_text())
        Path(raw_config["output_dir"]).mkdir(exist_ok=True, parents=True)
        config = Config.model_validate(raw_config)
    else:
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        config = Config(
            question=question,
            input_file=input_file,
            output_dir=output_dir,
            model=model,
            find_prompt=find_prompt,
            answer_prompt=answer_prompt,
        )

    logger.info("Loading and converting to sections")
    document_to_sections_dir(config.input_file, config.output_dir)
    logger.success("Done")

    logger.info("Loading Model")
    model = load_llama_cpp_model(config.model)
    logger.success("Done")

    logger.info("Answering")
    answer, sections_checked = find_retrieve_answer(
        model=model,
        sections_dir=config.output_dir,
        question=config.question,
        find_prompt=config.find_prompt,
        answer_prompt=config.answer_prompt,
    )
    logger.success("Done")

    logger.info("Sections checked:")
    logger.info(sections_checked)
    logger.info("Answer:")
    logger.info(answer)


def main():
    Fire(structured_qa)
