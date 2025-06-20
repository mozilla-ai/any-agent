import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from any_agent.tracing.agent_trace import AgentTrace

MAX_EVIDENCE_LENGTH: int = 50


class AgentEvaluationTools:
    def __init__(self, trace: AgentTrace):
        self.trace = trace

    def get_all_tools(self) -> list[Callable[[], Any]]:
        """Get all tool functions from this class.

        Returns:
            list[callable]: List of all tool functions

        """
        # Get all methods that don't start with underscore and aren't get_all_tools
        tools = []
        for attr_name in dir(self):
            if not attr_name.startswith("_") and attr_name != "get_all_tools":
                attr = getattr(self, attr_name)
                if callable(attr) and attr_name not in ["trace"]:
                    tools.append(attr)
        return tools

    def get_final_output(self) -> str | BaseModel | None:
        """Get the final output from the agent trace.

        Returns:
            str | BaseModel | None: The final output of the agent

        """
        return self.trace.final_output

    def get_tokens_used(self) -> int:
        """Get the number of tokens used by the agent as reported by the trace.

        Returns:
            int: The number of tokens used by the agent

        """
        return self.trace.tokens.total_tokens

    def get_number_of_steps(self) -> int:
        """Get the number of steps taken by the agent as reported by the trace.

        Returns:
            int: The number of steps taken by the agent

        """
        return len(self.trace.spans)

    def get_evidence_from_spans(self) -> str:
        """Get a summary of what happened in each step/span of the agent trace.

        This includes information about the input, output, and tool calls for each step.

        Returns:
            str: The evidence of all the spans in the trace

        """
        evidence = ""
        for idx, span in enumerate(self.trace.spans):
            evidence += (
                f"### Step {idx}: {span.attributes.get('gen_ai.operation.name')}\n"
            )
            if idx == 0:
                input_val = span.attributes.get("gen_ai.input.messages")
                # messages should always be json
                if input_val:
                    input_json = json.loads(input_val)
                    evidence += f"Input: {json.dumps(input_json, indent=2)}\n\n"

            tool_args = span.attributes.get("gen_ai.tool.args")
            if tool_args:
                args_json = json.loads(tool_args)
                tool_name = span.attributes.get("gen_ai.tool.name")
                evidence += f"Tool called: {tool_name}\n\n"
                evidence += f"Tool arguments: {json.dumps(args_json, indent=2)}\n\n"

            output = span.attributes.get("gen_ai.output")
            if output:
                try:
                    output_json = json.loads(output)
                    # the output can be quite long, truncate if needed
                    pretty_output = json.dumps(output_json, indent=2)
                    pretty_output = (
                        pretty_output[:MAX_EVIDENCE_LENGTH] + "...[TRUNCATED]"
                        if len(pretty_output) > MAX_EVIDENCE_LENGTH
                        else pretty_output
                    )
                    evidence += f"Output: {pretty_output}\n\n"
                except json.JSONDecodeError:
                    evidence += f"Output: {output}\n\n"
        return evidence
