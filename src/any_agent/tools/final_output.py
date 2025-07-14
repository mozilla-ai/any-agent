import json
from collections.abc import Callable

from pydantic import BaseModel, ValidationError


def prepare_final_output(
    output_type: type[BaseModel], instructions: str | None = None
) -> tuple[str, Callable[[str], dict[str, str | bool]]]:
    """Prepare instructions and tools for structured output, returning the function directly.

    Args:
        output_type: The Pydantic model type for structured output
        instructions: Original instructions to modify

    Returns:
        Tuple of (modified_instructions, final_output_function)

    """
    tool_name = "final_output"
    modified_instructions = instructions or ""
    modified_instructions += (
        f"You must call the {tool_name} tool when finished."
        f"The 'answer' argument passed to the {tool_name} tool must be a JSON string that matches the following schema:\n"
        f"{output_type.model_json_schema()}"
    )

    def final_output_tool(answer: str) -> dict[str, str | bool]:
        # First check if it's valid JSON
        try:
            json.loads(answer)
        except json.JSONDecodeError as json_err:
            return {
                "success": False,
                "result": f"Invalid JSON format: {json_err}. Please fix the 'answer' parameter so that it is a valid JSON string and call this tool again.",
            }
        # Then validate against the Pydantic model
        try:
            output_type.model_validate_json(answer)
        except ValidationError as e:
            return {
                "success": False,
                "result": f"Please fix this validation error: {e}. The format must conform to {output_type.model_json_schema()}",
            }
        else:
            return {"success": True, "result": answer}

    # Set the function name and docstring
    final_output_tool.__name__ = tool_name
    final_output_tool.__doc__ = f"""This tool is used to validate the final output. It must be called when the final answer is ready in order to ensure that the output is valid.

    Args:
        answer: The final output that can be loaded as a Pydantic model. This must be a JSON compatible string that matches the following schema:
            {output_type.model_json_schema()}

    Returns:
        A dictionary with the following keys:
            - success: True if the output is valid, False otherwise.
            - result: The final output if success is True, otherwise an error message.

    """

    return modified_instructions, final_output_tool
