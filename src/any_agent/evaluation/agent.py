import json
from typing import Any

from any_agent import AgentConfig, AnyAgent, TracingConfig
from any_agent.tracing.agent_trace import AgentTrace

MAX_EVIDENCE_LENGTH: int = 500


class AgentTooling:
    def __init__(self, trace: AgentTrace):
        self.trace = trace

    def get_final_output(self) -> str | None | dict[str, Any]:
        """Get the final output from the agent trace.

        Returns:
            str | None | dict: The final output of the agent

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


def get_agent(trace: AgentTrace, model: str) -> AnyAgent:
    tooling = AgentTooling(trace)

    instructions = """You are a helpful assistant that will be used to evaluate the correctness of an agent trace. Given a specific question regarding the quality of the something about the agent, utilize the appropriate tools in order to gather the answer needed in order to accurately answer the question. If you're asked anything about what the agent did, you should strongly consider using the get_evidence_from_spans tool to get the evidence. However, if the question is about specific details of the agent's actions, you don't necessarily need to use the get_evidence_from_spans tool.

    Answer with:
    1. "passed": true or false
    2. "reasoning": Brief explanation for your decision

    Your final answer should be the following valid JSON:

    ```json
    {
        "passed": <true or false>,
        "reasoning": "The answer to the question",
    }
    ```
    """
    agent_config = AgentConfig(
        model_id=model,
        instructions=instructions,
        tools=[
            tooling.get_final_output,
            tooling.get_tokens_used,
            tooling.get_number_of_steps,
            tooling.get_evidence_from_spans,
        ],
    )

    return AnyAgent.create(
        "tinyagent",
        agent_config=agent_config,
        tracing=TracingConfig(console=False),
    )
