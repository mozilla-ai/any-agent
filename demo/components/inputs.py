import copy
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from constants import (
    DEFAULT_EVALUATION_CRITERIA,
    DEFAULT_EVALUATION_MODEL,
    MODEL_OPTIONS,
)
from pydantic import BaseModel, ConfigDict

from any_agent import AgentFramework


class UserInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    location: str
    max_driving_hours: int
    date: datetime
    framework: str
    evaluation_model: str
    evaluation_criteria: list[dict[str, str]]
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
        time = st.selectbox(
            "Select a time",
            [datetime.strptime(f"{i:02d}:00", "%H:%M").time() for i in range(24)],
            index=9,
        )
    date = datetime.combine(date, time)

    supported_frameworks = [framework for framework in AgentFramework]

    framework = st.selectbox(
        "Select the agent framework to use",
        supported_frameworks,
        index=2,
        format_func=lambda x: x.name,
    )

    model_id = st.selectbox(
        "Select the model to use",
        MODEL_OPTIONS,
        index=1,
        format_func=lambda x: "/".join(x.split("/")[-3:]),
    )

    with st.expander("Custom Evaluation"):
        evaluation_model_id = st.selectbox(
            "Select the model to use for LLM-as-a-Judge evaluation",
            MODEL_OPTIONS,
            index=2,
            format_func=lambda x: "/".join(x.split("/")[-3:]),
        )

        evaluation_criteria = copy.deepcopy(DEFAULT_EVALUATION_CRITERIA)

        criteria_df = pd.DataFrame(evaluation_criteria)
        criteria_df = st.data_editor(
            criteria_df,
            column_config={
                "criteria": st.column_config.TextColumn(label="Criteria"),
            },
            hide_index=True,
            num_rows="dynamic",
        )

        new_criteria = []

        if len(criteria_df) > 20:
            st.error("You can only add up to 20 criteria for the purpose of this demo.")
            criteria_df = criteria_df[:20]

        for _, row in criteria_df.iterrows():
            if row["criteria"] == "":
                continue
            try:
                if len(row["criteria"].split(" ")) > 100:
                    msg = "Criteria is too long"
                    raise ValueError(msg)
                new_criteria.append({"criteria": row["criteria"]})
            except Exception as e:
                st.error(f"Error creating criterion: {e}")

    return UserInputs(
        model_id=model_id,
        location=location,
        max_driving_hours=max_driving_hours,
        date=date,
        framework=framework,
        evaluation_model=evaluation_model_id,
        evaluation_criteria=new_criteria,
        run_evaluation=st.checkbox("Run Evaluation", value=True),
    )
