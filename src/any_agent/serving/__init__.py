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

    __all__ = [
        "A2AServingConfig",
        "MCPServingConfig",
        "_get_a2a_app",
        "_get_a2a_app_async",
        "serve_a2a",
        "serve_a2a_async",
        "serve_mcp",
        "serve_mcp_async",
    ]
except ImportError:
    msg = "You need to `pip install 'any-agent[a2a]'` to use this method."

    class A2AServingConfig:  # type: ignore[no-redef]
        """Fail import."""

        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            """Fail import."""
            raise ImportError(msg) from None

    def _get_a2a_app(agent, serving_config):  # type: ignore[no-untyped-def,misc]
        raise ImportError(msg) from None

    async def _get_a2a_app_async(agent, serving_config):  # type: ignore[no-untyped-def,misc]
        raise ImportError(msg) from None

    def serve_a2a(  # type: ignore[no-untyped-def,misc]
        server,
        host,
        port,
        endpoint,
        log_level,
        server_queue,
    ):
        """Fail import."""
        raise ImportError(msg) from None

    def serve_a2a_async(  # type: ignore[no-untyped-def,misc]
        server,
        host,
        port,
        endpoint,
        log_level,
        server_queue,
    ):
        """Fail import."""
        raise ImportError(msg) from None
