from pydantic import BaseModel, ConfigDict


class MCPServingConfig(BaseModel):
    """Configuration for serving an agent using the Model Context Protocol (MCP).

    Example:
        config = MCPServingConfig(
            port=8080,
            endpoint="/my-agent",
        )

    """

    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    """Will be passed as argument to `uvicorn.run`."""

    port: int = 5000
    """Will be passed as argument to `uvicorn.run`."""

    endpoint: str = "/"
    """Will be pass as argument to `Starlette().add_route`"""

    log_level: str = "warning"
    """Will be passed as argument to the `uvicorn` server."""

    version: str = "0.1.0"
