from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from any_agent.tracing.agent_trace import AgentTrace


class TraceTools:
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

    def get_final_output(self) -> str | BaseModel | dict[str, Any] | None:
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

    def get_steps_taken(self) -> int:
        """Get the number of steps taken by the agent as reported by the trace.

        Returns:
            int: The number of steps taken by the agent

        """
        return len(self.trace.spans)

    def get_messages_from_trace(self) -> str:
        """Get a summary of what happened in each step/span of the agent trace.

        This includes information about the input, output, and tool calls for each step.

        Returns:
            str: The evidence of all the spans in the trace

        """
        messages = self.trace.spans_to_messages()
        evidence = ""
        for message in messages:
            evidence += f"### {message.role}\n{message.content}\n\n"
        return evidence

    def get_duration(self) -> float:
        """Get the duration of the agent trace.

        Returns:
            float: The duration in seconds of the agent trace

        """
        return self.trace.duration.total_seconds()
