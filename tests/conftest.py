import pytest
import rich.console

from any_agent.config import AgentFramework


@pytest.fixture(autouse=True)
def disable_rich_console(
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
) -> None:
    original_init = rich.console.Console.__init__

    def quiet_init(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if pytestconfig.option.capture != "no":
            kwargs["quiet"] = True
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(rich.console.Console, "__init__", quiet_init)


@pytest.fixture(params=list(AgentFramework), ids=lambda x: x.name)
def agent_framework(request: pytest.FixtureRequest) -> AgentFramework:
    return request.param  # type: ignore[no-any-return]
