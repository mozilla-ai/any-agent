import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import any_agent.tools.mcp.mcp_client as mcp_client_module
from any_agent.config import AgentFramework, MCPStdio
from any_agent.tools.mcp.mcp_client import MCPClient

MCP_CLIENT_PATH = Path(mcp_client_module.__file__)


class _BlockingFinder:
    """Meta path finder that makes a single module unimportable."""

    def __init__(self, blocked: str) -> None:
        self.blocked = blocked

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> None:
        """Raise for the blocked module, defer to the next finder otherwise."""
        if fullname == self.blocked:
            msg = f"No module named {fullname!r}"
            raise ImportError(msg)


def _import_with_blocked_module(blocked: str) -> ModuleType:
    """Load a fresh copy of mcp_client while `blocked` cannot be imported."""
    finder = _BlockingFinder(blocked)
    saved = {name: mod for name, mod in sys.modules.items() if name == blocked}
    for name in saved:
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    try:
        spec = importlib.util.spec_from_file_location(
            "isolated_mcp_client", MCP_CLIENT_PATH
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Pydantic resolves the deferred annotations through sys.modules, so the
        # isolated copy has to be registered while it is being built.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            module.MCPClient.model_rebuild()
        finally:
            del sys.modules[spec.name]
        return module
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(saved)


def test_module_imports_when_an_mcp_symbol_is_unavailable() -> None:
    """A missing mcp submodule must not break import of the client module.

    Regression test: the annotations in the class body used to be evaluated at
    import time, so a failed `mcp` import surfaced as `NameError: MCPTool` while
    importing `any_agent` itself instead of the intended deferred ImportError.
    """
    module = _import_with_blocked_module("mcp.client.streamable_http")

    assert module.missing_mcp_error is not None
    assert isinstance(module.missing_mcp_error, ImportError)
    assert module.MCPClient is not None


def test_construction_raises_friendly_error_when_mcp_is_unavailable() -> None:
    module = _import_with_blocked_module("mcp.client.streamable_http")

    with pytest.raises(ImportError, match="to use MCP"):
        module.MCPClient(
            config=MCPStdio(command="test", args=[]),
            framework=AgentFramework.OPENAI,
        )


def test_construction_succeeds_when_mcp_is_available() -> None:
    assert mcp_client_module.missing_mcp_error is None

    client = MCPClient(
        config=MCPStdio(command="test", args=[]),
        framework=AgentFramework.OPENAI,
    )

    assert client.framework is AgentFramework.OPENAI
