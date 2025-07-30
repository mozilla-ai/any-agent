import copy
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from constants import DEFAULT_EVALUATION_CASES, MODEL_OPTIONS
from pydantic import BaseModel, ConfigDict

from any_agent import AgentFramework


class UserInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    location: str
    max_driving_hours: int
    date: datetime
    framework: str
    evaluation_cases: list[str]
    llm_judge: str
    run_evaluation: bool


@st.cache_resource
def get_area(area_name: str) -> dict:
    """Get the area from Nominatim.

    Uses the [Nominatim API](https://nominatim.org/release-docs/develop/api/Search/).

    Args:
        area_name (str): The name of the area.

    Returns:
        dict: The area found.

    """
    response = requests.get(
        f"https://nominatim.openstreetmap.org/search?q={area_name}&format=jsonv2",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=5,
    )
    response.raise_for_status()
    return json.loads(response.content.decode())


def get_user_inputs() -> UserInputs:
    default_val = "Los Angeles California, US"

    location = st.text_input("Enter a location", value=default_val)
    if location:
        location_check = get_area(location)
        if not location_check:
            st.error("❌ Invalid location")

    max_driving_hours = st.number_input(
        "Enter the maximum driving hours", min_value=1, value=2
    )

    col_date, col_time = st.columns([2, 1])
    with col_date:
        date = st.date_input(
            "Select a date in the future", value=datetime.now() + timedelta(days=1)
        )
    with col_time:
        # default to 9am
        time = st.selectbox(
            "Select a time",
            [datetime.strptime(f"{i:02d}:00", "%H:%M").time() for i in range(24)],
            index=9,
        )
    date = datetime.combine(date, time)

    framework = st.selectbox(
        "Select the agent framework to use",
        list(AgentFramework),
        index=2,
        format_func=lambda x: x.name,
    )

    model_id = st.selectbox(
        "Select the model to use",
        MODEL_OPTIONS,
        index=1,
        format_func=lambda x: "/".join(x.split("/")[-3:]),
    )

    # Add evaluation case section
    with st.expander("Custom Evaluation"):
        evaluation_model_id = st.selectbox(
            "Select the model to use for LLM-as-a-Judge evaluation",
            MODEL_OPTIONS,
            index=2,
            format_func=lambda x: "/".join(x.split("/")[-3:]),
        )
        evaluation_cases = copy.deepcopy(DEFAULT_EVALUATION_CASES)
        llm_judge = evaluation_model_id

        evaluation_cases_df = pd.DataFrame({"case": evaluation_cases})
        evaluation_cases_df = st.data_editor(
            evaluation_cases_df,
            column_config={
                "case": st.column_config.TextColumn(label="Evaluation Case"),
            },
            hide_index=True,
            num_rows="dynamic",
        )

        if len(evaluation_cases_df) > 20:
            st.error("You can only add up to 20 cases for the purpose of this demo.")
            evaluation_cases_df = evaluation_cases_df[:20]

    return UserInputs(
        model_id=model_id,
        location=location,
        max_driving_hours=max_driving_hours,
        date=date,
        framework=framework,
        evaluation_cases=list(evaluation_cases_df["case"]),
        llm_judge=llm_judge,
        run_evaluation=st.checkbox("Run Evaluation", value=True),
    )
