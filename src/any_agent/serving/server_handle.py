from __future__ import annotations

import asyncio
from typing import Any

import uvicorn
from pydantic import BaseModel, Field

from any_agent.logging import logger


class ServerHandle(BaseModel):
    """A handle for managing an async server instance.

    This class provides a clean interface for managing the lifecycle of a server
    without requiring manual management of the underlying task and server objects.
    """

    task: asyncio.Task[Any]
    server: uvicorn.Server

    class Config:
        arbitrary_types_allowed = True

    async def shutdown(self, timeout_seconds: float = 10.0) -> None:
        """Gracefully shutdown the server with a timeout.

        Args:
            timeout_seconds: Maximum time to wait for graceful shutdown before forcing cancellation.

        """
        if not self.is_running():
            return  # Already shut down

        self.server.should_exit = True
        try:
            await asyncio.wait_for(self.task, timeout=timeout_seconds)
        except TimeoutError:
            logger.warning(
                f"Server shutdown timed out after {timeout_seconds}s, forcing cancellation"
            )
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            logger.error(f"Error during server shutdown: {e}")
            # Still try to cancel the task to clean up
            if not self.task.done():
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass

    def is_running(self) -> bool:
        """Check if the server is still running.

        Returns:
            True if the server task is still running, False otherwise.

        """
        return not self.task.done()

    @property
    def port(self) -> int:
        """Get the port the server is running on.

        Returns:
            The port number the server is running on.

        """
        port = self.server.servers[0].sockets[0].getsockname()[1]
        assert port is not None
        assert isinstance(port, int)
        return port
