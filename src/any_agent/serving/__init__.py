from .config_mcp import MCPServingConfig
from .server_mcp import (
    serve_mcp,
    serve_mcp_async,
)

__all__ = [
    "MCPServingConfig",
    "serve_mcp",
    "serve_mcp_async",
]

try:
    from .config_a2a import A2AServingConfig
    from .server_a2a import (
        _get_a2a_app,
        _get_a2a_app_async,
        serve_a2a,
        serve_a2a_async,
    )

    __all__ += [
        "A2AServingConfig",
        "_get_a2a_app",
        "_get_a2a_app_async",
        "serve_a2a",
        "serve_a2a_async",
    ]
except ImportError:
    msg = "You need to `pip install 'any-agent[a2a]'` to use this method."
    raise ImportError(msg) from None
