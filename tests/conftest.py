import pytest
import rich.console

from any_agent.config import AgentFramework


@pytest.fixture(autouse=True)
def disable_rich_console(monkeypatch, pytestconfig):
    original_init = rich.console.Console.__init__

    def quiet_init(self, *args, **kwargs):
        if pytestconfig.option.capture != "no":
            kwargs["quiet"] = True
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(rich.console.Console, "__init__", quiet_init)


@pytest.fixture(params=AgentFramework)
def agent_framework(request: pytest.FixtureRequest) -> AgentFramework:
    return request.param
