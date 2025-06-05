from collections.abc import Callable

from pydantic import BaseModel, ValidationError


def _create_final_output_tool(
    output_type: type[BaseModel],
) -> Callable[[str], dict]:  # type: ignore[type-arg]
    def final_output(answer: str) -> dict:  # type: ignore[type-arg]
        try:
            output_type.model_validate_json(answer)
        except ValidationError as e:
            return {
                "success": False,
                "result": f"Please fix this validation error: {e}. The format must conform to {output_type.model_json_schema()}",
            }
        else:
            return {"success": True, "result": answer}

    final_output.__doc__ = f"""You must call this tool in order to return the final answer.

        Args:
            final_output: The final output that can be loaded as a Pydantic model. This must be a JSON compatible string that matches the following schema:
                {output_type.model_json_schema()}

        Returns:
            A dictionary with the following keys:
                - success: True if the output is valid, False otherwise.
                - result: The final output if success is True, otherwise an error message.

        """
    return final_output
