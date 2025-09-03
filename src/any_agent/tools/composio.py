# mypy: disable-error-code="attr-defined,operator,type-arg"
import inspect as i
import typing as t
from types import FunctionType

try:
    from composio.client.types import Tool
    from composio.core.provider import AgenticProvider
    from composio.core.provider.agentic import AgenticProviderExecuteFn
except ImportError as e:
    msg = "Composio is not installed. Please install it with `pip install composio`."
    raise ImportError(msg) from e


TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _get_parameters(tool: Tool) -> list[i.Parameter]:
    parameters = []
    if tool.input_parameters and isinstance(tool.input_parameters, dict):
        properties = tool.input_parameters.get("properties", {})
        required = tool.input_parameters.get("required", [])

        for param_name, param_info in properties.items():
            base_param_type = TYPE_MAPPING.get(param_info.get("type", "string"), str)

            if param_name not in required:
                param = i.Parameter(
                    param_name,
                    i.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=base_param_type | None,
                )
            else:
                param = i.Parameter(
                    param_name,
                    i.Parameter.KEYWORD_ONLY,
                    annotation=base_param_type,
                )
            parameters.append(param)

    return parameters


class CallableProvider(AgenticProvider[t.Callable, list[t.Callable]], name="callable"):
    """Composio toolset for generic callables."""

    __schema_skip_defaults__ = True

    def wrap_tool(
        self,
        tool: Tool,
        execute_tool: AgenticProviderExecuteFn,
    ) -> t.Callable:
        """Wrap composio tool as a python callable."""
        docstring = tool.description
        docstring += "\nArgs:"
        for _param, _schema in tool.input_parameters["properties"].items():
            docstring += "\n    "
            docstring += _param + ": " + _schema.get("description", _param.title())

        docstring += "\nReturns:"
        docstring += "\n    A dictionary containing response from the action"

        def _execute(**kwargs: t.Any) -> dict:
            return execute_tool(slug=tool.slug, arguments=kwargs)

        function = FunctionType(
            code=_execute.__code__,
            name=tool.slug,
            globals=globals(),
            closure=_execute.__closure__,
        )

        parameters = _get_parameters(tool)
        function.__annotations__ = {p.name: p.annotation for p in parameters} | {
            "return": dict
        }
        function.__signature__ = i.Signature(
            parameters=parameters, return_annotation=dict
        )
        function.__doc__ = docstring
        return function

    def wrap_tools(
        self,
        tools: t.Sequence[Tool],
        execute_tool: AgenticProviderExecuteFn,
    ) -> list[t.Callable]:
        """Wrap composio tools as python functions."""
        return [self.wrap_tool(tool, execute_tool) for tool in tools]
