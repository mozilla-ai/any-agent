import json

import streamlit as st
from components.agent_status import export_logs
from components.inputs import UserInputs
from constants import DEFAULT_TOOLS
from config import Config

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.evaluation import LlmJudge
from any_agent.evaluation.schemas import EvaluationOutput
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.tracing.attributes import GenAI
from any_agent.tracing.otel_types import StatusCode


async def display_evaluation_results(results: list[EvaluationOutput]):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Criteria Results")
        for i, result in enumerate(results):
            if result.passed:
                st.success(f"✅ Criterion {i + 1}")
            else:
                st.error(f"❌ Criterion {i + 1}")
            st.write(f"**Reasoning:** {result.reasoning}")

    with col2:
        st.markdown("#### Overall Score")
        total_criteria = len(results)
        passed_criteria = sum(1 for result in results if result.passed)

        st.markdown(f"### {passed_criteria}/{total_criteria}")
        percentage = (
            (passed_criteria / total_criteria) * 100 if total_criteria > 0 else 0
        )
        st.progress(percentage / 100)
        st.markdown(f"**{percentage:.1f}%**")


async def evaluate_agent(
    config: Config, agent_trace: AgentTrace
) -> list[EvaluationOutput]:
    st.markdown("### 📊 Evaluation Results")

    with st.spinner("Evaluating results..."):
        results = []

        judge = LlmJudge(model_id=config.evaluation_model, framework=config.framework)

        for i, criterion in enumerate(config.evaluation_criteria):
            context = f"Agent Trace:\n{agent_trace.model_dump_json(indent=2)}"

            result = await judge.run_async(
                context=context, question=criterion["criteria"]
            )
            results.append(result)

            st.write(f"Evaluated criterion {i + 1}/{len(config.evaluation_criteria)}")

    return results


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
        evaluation_model=user_inputs.evaluation_model,
        evaluation_criteria=user_inputs.evaluation_criteria,
    )

    agent = await AnyAgent.create_async(
        agent_framework=config.framework,
        agent_config=config.main_agent,
    )
    return agent, config


async def display_output(agent_trace: AgentTrace):
    with st.expander("### 🧩 Agent Trace"):
        for span in agent_trace.spans:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{span.name}**")
                if span.attributes:
                    if GenAI.INPUT_MESSAGES in span.attributes:
                        try:
                            input_value = json.loads(
                                span.attributes[GenAI.INPUT_MESSAGES]
                            )
                            if isinstance(input_value, list) and len(input_value) > 0:
                                st.write(f"Input: {input_value[-1]}")
                            else:
                                st.write(f"Input: {input_value}")
                        except Exception:
                            st.write(f"Input: {span.attributes[GenAI.INPUT_MESSAGES]}")

                    if GenAI.TOOL_ARGS in span.attributes:
                        try:
                            tool_args = json.loads(span.attributes[GenAI.TOOL_ARGS])
                            st.write(f"Tool Args: {tool_args}")
                        except Exception:
                            st.write(f"Tool Args: {span.attributes[GenAI.TOOL_ARGS]}")

                    if GenAI.OUTPUT in span.attributes:
                        try:
                            output_value = json.loads(span.attributes[GenAI.OUTPUT])
                            if isinstance(output_value, list) and len(output_value) > 0:
                                st.write(f"Output: {output_value[-1]}")
                            else:
                                st.write(f"Output: {output_value}")
                        except Exception:
                            st.write(f"Output: {span.attributes[GenAI.OUTPUT]}")
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
            st.info(f"📦 Total Tokens: {agent_trace.tokens.total_tokens:,}")
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

        def update_status(message: str):
            status.update(label=message, expanded=False, state="running")

        export_logs(agent, update_status)
        agent_trace: AgentTrace = await agent.run_async(query, **kwargs)
        status.update(label="Finished!", expanded=False, state="complete")
        return agent_trace
