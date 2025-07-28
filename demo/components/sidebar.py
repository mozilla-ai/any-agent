import streamlit as st

from components.inputs import UserInputs, get_user_inputs


def ssf_sidebar() -> UserInputs:
    st.markdown("### Configuration")
    st.markdown("Built using [Any-Agent](https://github.com/mozilla-ai/any-agent)")
    return get_user_inputs()
