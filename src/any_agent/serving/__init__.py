from .mcp.config_mcp import MCPServingConfig
from .mcp.server_mcp import (
    serve_mcp_async,
)
from .server_handle import ServerHandle

__all__ = [
    "MCPServingConfig",
    "ServerHandle",
    "serve_mcp_async",
]

try:
    from .a2a.config_a2a import A2AServingConfig
    from .a2a.server_a2a import (
        _get_a2a_app_async,
        serve_a2a_async,
    )

    __all__ += [
        "A2AServingConfig",
        "_get_a2a_app_async",
        "serve_a2a_async",
    ]
except ImportError:

    def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        msg = "You need to `pip install 'any-agent[a2a]'` to use this method."
        raise ImportError(msg)

    A2AServingConfig = _raise  # type: ignore[assignment,misc]
    _get_a2a_app_async = _raise
    serve_a2a_async = _raise
    __all__ += [
        "A2AServingConfig",
        "_get_a2a_app_async",
        "serve_a2a_async",
    ]
