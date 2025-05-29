import asyncio
from unittest.mock import patch

from a2a.types import AgentCapabilities, AgentCard

from any_agent.tools.a2a import a2a_tool


def test_a2a_tool_name_default():
    fun_name = "some_name"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = AgentCard(
            capabilities=AgentCapabilities(),
            defaultInputModes=[],
            defaultOutputModes=[],
            description="dummy",
            name=fun_name,
            skills=[],
            url="http://example.com/test",
            version="0.0.1",
        )
        created_fun = asyncio.run(a2a_tool("http://example.com/test"))
        assert created_fun.__name__ == f"call_{fun_name}"


def test_a2a_tool_name_specific():
    other_name = "other_name"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = AgentCard(
            capabilities=AgentCapabilities(),
            defaultInputModes=[],
            defaultOutputModes=[],
            description="dummy",
            name="some_name",
            skills=[],
            url="http://example.com/test",
            version="0.0.1",
        )
        created_fun = asyncio.run(a2a_tool("http://example.com/test", other_name))
        assert created_fun.__name__ == f"call_{other_name}"


def test_a2a_tool_name_whitespace():
    fun_name = "  some_n  ame  "
    corrected_fun_name = "some_n_ame"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = AgentCard(
            capabilities=AgentCapabilities(),
            defaultInputModes=[],
            defaultOutputModes=[],
            description="dummy",
            name=fun_name,
            skills=[],
            url="http://example.com/test",
            version="0.0.1",
        )
        created_fun = asyncio.run(a2a_tool("http://example.com/test"))
        assert created_fun.__name__ == f"call_{corrected_fun_name}"


def test_a2a_tool_name_exotic_whitespace():
    fun_name = " \n so \t me_n\t ame  \n"
    corrected_fun_name = "so_me_n_ame"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = AgentCard(
            capabilities=AgentCapabilities(),
            defaultInputModes=[],
            defaultOutputModes=[],
            description="dummy",
            name=fun_name,
            skills=[],
            url="http://example.com/test",
            version="0.0.1",
        )
        created_fun = asyncio.run(a2a_tool("http://example.com/test"))
        assert created_fun.__name__ == f"call_{corrected_fun_name}"


def test_a2a_tool_name_specific_whitespace():
    other_name = " \n oth \t er_n\t ame  \n"
    corrected_other_name = "oth_er_n_ame"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = AgentCard(
            capabilities=AgentCapabilities(),
            defaultInputModes=[],
            defaultOutputModes=[],
            description="dummy",
            name="some_name",
            skills=[],
            url="http://example.com/test",
            version="0.0.1",
        )
        created_fun = asyncio.run(a2a_tool("http://example.com/test", other_name))
        assert created_fun.__name__ == f"call_{corrected_other_name}"
