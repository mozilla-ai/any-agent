"""Retry utilities for handling transient failures in agent operations."""

import asyncio
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_exception: Exception) -> None:
        """Initialize RetryError.

        Args:
            attempts: Number of attempts made
            last_exception: The last exception that was raised

        """
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Failed after {attempts} attempts. Last error: {last_exception}"
        )


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function with retry logic

    Example:
        ```python
        @retry_with_backoff(max_attempts=3, initial_delay=1.0)
        def call_api():
            # API call that might fail
            pass
        ```

    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func_name = getattr(func, "__name__", repr(func))

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.exception(
                            "All %d retry attempts failed for %s",
                            max_attempts,
                            func_name,
                        )
                        raise RetryError(max_attempts, e) from e

                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.2fs...",
                        attempt,
                        max_attempts,
                        func_name,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)

            # This should never be reached, but satisfies type checker
            if last_exception:
                raise RetryError(max_attempts, last_exception)
            msg = "Unexpected retry state"
            raise RuntimeError(msg)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc,no-any-return]
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.exception(
                            "All %d retry attempts failed for %s",
                            max_attempts,
                            func_name,
                        )
                        raise RetryError(max_attempts, e) from e

                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.2fs...",
                        attempt,
                        max_attempts,
                        func_name,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)

            # This should never be reached, but satisfies type checker
            if last_exception:
                raise RetryError(max_attempts, last_exception)
            msg = "Unexpected retry state"
            raise RuntimeError(msg)

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper

    return decorator
