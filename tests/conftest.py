import sys

import pytest
import rich.console


@pytest.fixture
def refresh_tools() -> None:
    """Smolagents tool wrapping hacks the original function signature
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
