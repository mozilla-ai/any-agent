import sys

import pytest


@pytest.fixture(scope="function")
def refresh_tools():
    """
    smolagents tool wrapping hacks the original function signature
    of the tool that you pass.
    That causes that a tool already wrapped in the same python
    process will fail the second time you try to wrap it.
    This is the simplest way I found to invalidate the modified
    signature.
    """
    tool_modules = []
    for k in sys.modules.keys():
        if "any_agent.tools" in k:
            tool_modules.append(k)
    for module in tool_modules:
        del sys.modules[module]
