import json

import streamlit as st
from components.agent_status import export_logs
from components.inputs import UserInputs
from constants import DEFAULT_TOOLS
from surf_spot_finder.config import Config

from any_agent import AgentConfig, AgentFramework, AnyAgent, TracingConfig
from any_agent.evaluation import TraceEvaluationResult, evaluate
from any_agent.tracing.otel_types import StatusCode
from any_agent.tracing.trace import AgentSpan, AgentTrace


async def display_evaluation_results(result: TraceEvaluationResult):
    if result.ground_truth_result is not None:
        all_results = [*result.checkpoint_results, result.ground_truth_result]
    else:
        all_results = result.checkpoint_results

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Criteria Results")
        for checkpoint in all_results:
            if checkpoint.passed:
                st.success(f"✅ {checkpoint.criteria}")
            else:
                st.error(f"❌ {checkpoint.criteria}")

    with col2:
        st.markdown("#### Overall Score")
        total_points = sum([result.points for result in all_results])
        if total_points == 0:
            msg = "Total points is 0, cannot calculate score."
            raise ValueError(msg)
        passed_points = sum([result.points for result in all_results if result.passed])

        # Create a nice score display
        st.markdown(f"### {passed_points}/{total_points}")
        percentage = (passed_points / total_points) * 100
        st.progress(percentage / 100)
        st.markdown(f"**{percentage:.1f}%**")


async def evaluate_agent(
    config: Config, agent_trace: AgentTrace
) -> TraceEvaluationResult:
    assert len(config.evaluation_cases) == 1, (
        "Only one evaluation case is supported in the demo"
    )
    st.markdown("### 📊 Evaluation Results")

    with st.spinner("Evaluating results..."):
        case = config.evaluation_cases[0]
        result: TraceEvaluationResult = evaluate(
            evaluation_case=case,
            trace=agent_trace,
            agent_framework=config.framework,
        )
    return result


async def configure_agent(user_inputs: UserInputs) -> tuple[AnyAgent, Config]:
    if "huggingface" in user_inputs.model_id:
        model_args = {
            "extra_headers": {"X-HF-Bill-To": "mozilla-ai"},
            "temperature": 0.0,
        }
    else:
        model_args = {}

    if user_inputs.framework == AgentFramework.AGNO:
        agent_args = {"tool_call_limit": 20}
    else:
        agent_args = {}

    agent_config = AgentConfig(
        model_id=user_inputs.model_id,
        model_args=model_args,
        agent_args=agent_args,
        tools=DEFAULT_TOOLS,
    )

    config = Config(
        location=user_inputs.location,
        max_driving_hours=user_inputs.max_driving_hours,
        date=user_inputs.date,
        framework=user_inputs.framework,
        main_agent=agent_config,
        managed_agents=[],
        evaluation_cases=[user_inputs.evaluation_case],
    )

    agent = await AnyAgent.create_async(
        agent_framework=config.framework,
        agent_config=config.main_agent,
        managed_agents=config.managed_agents,
        tracing=TracingConfig(console=True, cost_info=True),
    )
    return agent, config


async def display_output(agent_trace: AgentTrace):
    # Display the agent trace in a more organized way
    with st.expander("### 🧩 Agent Trace"):
        for span in agent_trace.spans:
            # Header with name and status
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{span.name}**")
                if span.attributes:
                    # st.json(span.attributes, expanded=False)
                    if "input.value" in span.attributes:
                        try:
                            input_value = json.loads(span.attributes["input.value"])
                            if isinstance(input_value, list) and len(input_value) > 0:
                                st.write(f"Input: {input_value[-1]}")
                            else:
                                st.write(f"Input: {input_value}")
                        except Exception:
                            st.write(f"Input: {span.attributes['input.value']}")
                    if "output.value" in span.attributes:
                        try:
                            output_value = json.loads(span.attributes["output.value"])
                            if isinstance(output_value, list) and len(output_value) > 0:
                                st.write(f"Output: {output_value[-1]}")
                            else:
                                st.write(f"Output: {output_value}")
                        except Exception:
                            st.write(f"Output: {span.attributes['output.value']}")
            with col2:
                status_color = (
                    "green" if span.status.status_code == StatusCode.OK else "red"
                )
                st.markdown(
                    f"<span style='color: {status_color}'>● {span.status.status_code.name}</span>",
                    unsafe_allow_html=True,
                )

    with st.expander("### 🏄 Results", expanded=True):
        time_col, cost_col, tokens_col = st.columns(3)
        duration = agent_trace.duration.total_seconds()
        with time_col:
            st.info(f"⏱️ Execution Time: {duration:0.2f} seconds")
        with cost_col:
            st.info(f"💰 Estimated Cost: ${agent_trace.cost.total_cost:.6f}")
        with tokens_col:
            st.info(f"📦 Total Tokens: {agent_trace.usage.total_tokens:,}")
        st.markdown("#### Final Output")
        st.info(agent_trace.final_output)


async def run_agent(agent, config) -> AgentTrace:
    st.markdown("#### 🔍 Running Surf Spot Finder with query")

    query = config.input_prompt_template.format(
        LOCATION=config.location,
        MAX_DRIVING_HOURS=config.max_driving_hours,
        DATE=config.date,
    )

    st.code(query, language="text")
    kwargs = {}
    if (
        config.framework == AgentFramework.OPENAI
        or config.framework == AgentFramework.TINYAGENT
    ):
        kwargs["max_turns"] = 20
    elif config.framework == AgentFramework.SMOLAGENTS:
        kwargs["max_steps"] = 20
    if config.framework == AgentFramework.LANGCHAIN:
        from langchain_core.runnables import RunnableConfig

        kwargs["config"] = RunnableConfig(recursion_limit=20)
    elif config.framework == AgentFramework.GOOGLE:
        from google.adk.agents.run_config import RunConfig

        kwargs["run_config"] = RunConfig(max_llm_calls=20)

    with st.status("Agent is running...", expanded=False, state="running") as status:

        def update_span(span: AgentSpan):
            # Process input value
            input_value = span.attributes.get("input.value", "")
            if input_value:
                try:
                    parsed_input = json.loads(input_value)
                    if isinstance(parsed_input, list) and len(parsed_input) > 0:
                        input_value = str(parsed_input[-1])
                except Exception:
                    pass

            # Process output value
            output_value = span.attributes.get("output.value", "")
            if output_value:
                try:
                    parsed_output = json.loads(output_value)
                    if isinstance(parsed_output, list) and len(parsed_output) > 0:
                        output_value = str(parsed_output[-1])
                except Exception:
                    pass

            # Truncate long values
            max_length = 800
            if len(input_value) > max_length:
                input_value = f"[Truncated]...{input_value[-max_length:]}"
            if len(output_value) > max_length:
                output_value = f"[Truncated]...{output_value[-max_length:]}"

            # Create a cleaner message format
            if input_value or output_value:
                message = f"Step: {span.name}\n"
                if input_value:
                    message += f"Input: {input_value}\n"
                if output_value:
                    message += f"Output: {output_value}"
            else:
                message = f"Step: {span.name}\n{span}"

            status.update(label=message, expanded=False, state="running")

        export_logs(agent, update_span)
        agent_trace: AgentTrace = await agent.run_async(query, **kwargs)
        status.update(label="Finished!", expanded=False, state="complete")

        agent.exit()
        return agent_trace
