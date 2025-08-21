import asyncio

import nest_asyncio
import streamlit as st
from components.sidebar import ssf_sidebar
from constants import DEFAULT_TOOLS
from services.agent import (
    configure_agent,
    display_evaluation_results,
    display_output,
    evaluate_agent,
    run_agent,
)

nest_asyncio.apply()

st.set_page_config(page_title="Surf Spot Finder", page_icon="🏄", layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"] {
            max-width: 99% !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    user_inputs = ssf_sidebar()
    is_valid = user_inputs is not None
    run_button = st.button("Run Agent 🤖", disabled=not is_valid, type="primary")


async def main():
    if run_button:
        agent, agent_config = await configure_agent(user_inputs)
        agent_trace = await run_agent(agent, agent_config)

        await display_output(agent_trace)

        if user_inputs.run_evaluation:
            evaluation_results = await evaluate_agent(agent_config, agent_trace)
            await display_evaluation_results(evaluation_results)
    else:
        st.title("🏄 Surf Spot Finder")
        st.markdown(
            "Find the best surfing spots based on your location and preferences! [Github Repo](https://github.com/mozilla-ai/surf-spot-finder)"
        )
        st.info(
            "👈 Configure your search parameters in the sidebar and click Run to start!"
        )

        st.markdown("### 🛠️ Available Tools")

        st.markdown("""
        The AI Agent built for this project has a few tools available for use in order to find the perfect surf spot.
        The agent is given the freedom to use (or not use) these tools in order to accomplish the task.
        """)

        weather_tools = [
            tool
            for tool in DEFAULT_TOOLS
            if "forecast" in tool.__name__ or "weather" in tool.__name__
        ]
        for tool in weather_tools:
            with st.expander(f"🌤️ {tool.__name__}"):
                st.markdown(tool.__doc__ or "No description available")
        location_tools = [
            tool
            for tool in DEFAULT_TOOLS
            if "lat" in tool.__name__
            or "lon" in tool.__name__
            or "area" in tool.__name__
        ]
        for tool in location_tools:
            with st.expander(f"📍 {tool.__name__}"):
                st.markdown(tool.__doc__ or "No description available")

        web_tools = [
            tool
            for tool in DEFAULT_TOOLS
            if "web" in tool.__name__ or "search" in tool.__name__
        ]
        for tool in web_tools:
            with st.expander(f"🌐 {tool.__name__}"):
                st.markdown(tool.__doc__ or "No description available")

        if len(weather_tools) + len(location_tools) + len(web_tools) != len(
            DEFAULT_TOOLS
        ):
            st.warning(
                "Some tools are not listed. Please check the code for more details."
            )

        st.markdown("### 📊 Custom Evaluation")
        st.markdown("""
        The Surf Spot Finder includes a powerful evaluation system that allows you to customize how the agent's performance is assessed.
        You can find these settings in the sidebar under the "Custom Evaluation" expander.
        """)

        with st.expander("Learn more about Custom Evaluation"):
            st.markdown("""
            #### What is Custom Evaluation?
            The Custom Evaluation feature uses an LLM-as-a-Judge approach to evaluate how well the agent performs its task.
            An LLM will be given the complete agent trace (not just the final answer), and will assess the agent's performance based on the criteria you set.
            You can customize:

            - **Evaluation Model**: Choose which LLM should act as the judge
            - **Evaluation Criteria**: Define specific checkpoints that the agent should meet
            - **Scoring System**: Assign points to each criterion

            #### How to Use Custom Evaluation

            1. **Select an Evaluation Model**: Choose which LLM you want to use as the judge
            2. **Edit Checkpoints**: Use the data editor to:
               - Add new evaluation criteria
               - Modify existing criteria
               - Adjust point values
               - Remove criteria you don't want to evaluate

            #### Example Criteria
            You can evaluate things like:
            - Tool usage and success
            - Order of operations
            - Quality of final recommendations
            - Response completeness
            - Number of steps taken

            #### Tips for Creating Good Evaluation Criteria
            - Be specific about what you want to evaluate
            - Use clear, unambiguous language
            - Consider both process (how the agent works) and outcome (what it produces)
            - Assign appropriate point values based on importance

            The evaluation results will be displayed after each agent run, showing how well the agent met your custom criteria.
            """)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
