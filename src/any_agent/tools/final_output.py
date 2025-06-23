import json

from pydantic import BaseModel, ValidationError


class FinalOutputTool:
    """A serializable final output tool that avoids closure issues."""

    def __init__(self, output_type: type[BaseModel]):
        """Create the function that will be used as a tool."""
        self.output_type = output_type
        # Set docstring for the callable object
        self.__doc__ = f"""You must call this tool in order to return the final answer.

        Args:
            answer: The final output that can be loaded as a Pydantic model. This must be a JSON compatible string that matches the following schema:
                {output_type.model_json_schema()}

        Returns:
            A dictionary with the following keys:
                - success: True if the output is valid, False otherwise.
                - result: The final output if success is True, otherwise an error message.

        """
        # Set function name for tool frameworks that expect it
        self.__name__ = "final_output"

    def __call__(self, answer: str) -> dict:  # type: ignore[type-arg]
        """Validate the final output."""
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
            self.output_type.model_validate_json(answer)
        except ValidationError as e:
            return {
                "success": False,
                "result": f"Please fix this validation error: {e}. The format must conform to {self.output_type.model_json_schema()}",
            }
        else:
            return {"success": True, "result": answer}
