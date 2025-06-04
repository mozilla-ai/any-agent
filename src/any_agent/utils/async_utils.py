"""Utilities for handling async/sync context interactions."""

import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_async_in_sync_context(coro: Coroutine[Any, Any, T]) -> T:
    """Run async code in a sync context.

    This utility handles the common pattern of needing to run async code
    from a sync context, properly handling cases where an event loop
    is already running.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    """
    try:
        # Check if there's already an event loop running in this thread
        asyncio.get_running_loop()

        # If we get here, there IS a loop running, so we can't use asyncio.run()
        # directly because it would try to create a second loop in the same thread
        def run_in_thread() -> T:
            # This runs in a fresh thread with no event loop, so asyncio.run() is safe
            return asyncio.run(coro)

        # Execute the async code in a separate thread to avoid loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(run_in_thread).result()
    except RuntimeError:
        # No event loop running in this thread, so we can safely use asyncio.run()
        return asyncio.run(coro) 