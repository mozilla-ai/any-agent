import importlib
import re


def is_import(instructions):
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
    return bool(re.match(pattern, instructions))


def get_instructions(instructions: str | None) -> str | None:
    """Get the instructions from an external module.

    Args:
        instructions: Depending on the syntax used:

            - An import that points to a string in an external module.
                For example: `agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX`.
                The string will be imported from the external module.

            - A regular string containing instructions.
                For example: `You are a helpful assistant`.
                The string will be returned as is.

    Returns:
        Either the imported string or the input string as is.

    Raises:
        ValueError: If `instructions` is an import but doesn't point to a string.
    """
    if instructions and is_import(instructions):
        module, obj = instructions.rsplit(".", 1)
        module = importlib.import_module(module)
        imported = getattr(module, obj)
        if not isinstance(imported, str):
            raise ValueError(
                "Instructions were identified as an import"
                f" but the value imported is not a string:  {instructions}"
            )
        return imported
    return instructions
