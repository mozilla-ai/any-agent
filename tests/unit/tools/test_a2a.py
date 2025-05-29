import asyncio
from unittest.mock import MagicMock, patch

import pytest

from a2a.types import AgentCard, AgentCapabilities
from any_agent.tools.a2a_tool import a2a_tool

def test_a2a_tool_name_default():
    fun_name = "some_name"
    with (
        patch("any_agent.tools.a2a_tool.A2ACardResolver.get_agent_card") as agent_card_mock
    ):
        agent_card_mock.return_value = AgentCard(
            capabilities=AgentCapabilities(),
            defaultInputModes=[],
            defaultOutputModes=[],
            description="dummy",
            name=fun_name,
            skills=[],
            url="http://example.com/test",
            version="0.0.1"
        )
        created_fun = asyncio.run(a2a_tool("http://example.com/test"))
        assert created_fun.__name__ == f'call_{fun_name}'

def test_a2a_tool_name_specific():
    other_name = "other_name"
    with (
        patch("any_agent.tools.a2a_tool.A2ACardResolver.get_agent_card") as agent_card_mock
    ):
        agent_card_mock.return_value = AgentCard(
            capabilities=AgentCapabilities(),
            defaultInputModes=[],
            defaultOutputModes=[],
            description="dummy",
            name="some_name",
            skills=[],
            url="http://example.com/test",
            version="0.0.1"
        )
        created_fun = asyncio.run(a2a_tool("http://example.com/test", other_name))
        assert created_fun.__name__ == f'call_{other_name}'