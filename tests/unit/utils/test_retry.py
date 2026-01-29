"""Tests for retry utilities."""

import asyncio
import time
from unittest.mock import Mock

import pytest

from any_agent.utils.retry import RetryError, retry_with_backoff


class TestRetryWithBackoff:
    """Test suite for retry_with_backoff decorator."""

    def test_successful_first_attempt(self) -> None:
        """Test that function succeeds on first attempt without retry."""
        mock_func = Mock(return_value="success")
        decorated = retry_with_backoff()(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_successful_after_retries(self) -> None:
        """Test that function succeeds after some failed attempts."""
        mock_func = Mock(
            side_effect=[ValueError("fail"), ValueError("fail"), "success"]
        )
        decorated = retry_with_backoff(max_attempts=3, initial_delay=0.01)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_all_attempts_fail(self) -> None:
        """Test that RetryError is raised when all attempts fail."""
        mock_func = Mock(side_effect=ValueError("persistent error"))
        decorated = retry_with_backoff(max_attempts=3, initial_delay=0.01)(mock_func)

        with pytest.raises(RetryError) as exc_info:
            decorated()

        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, ValueError)
        assert str(exc_info.value.last_exception) == "persistent error"
        assert mock_func.call_count == 3

    def test_exponential_backoff_timing(self) -> None:
        """Test that delays follow exponential backoff pattern."""
        call_times = []

        @retry_with_backoff(max_attempts=3, initial_delay=0.1, exponential_base=2.0)
        def failing_func() -> None:
            call_times.append(time.time())
            raise ValueError("fail")

        with pytest.raises(RetryError):
            failing_func()

        # Verify we have 3 attempts
        assert len(call_times) == 3

        # Check delays are approximately correct (with some tolerance)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.08 < delay1 < 0.15  # ~0.1s
        assert 0.18 < delay2 < 0.25  # ~0.2s (0.1 * 2)

    def test_max_delay_cap(self) -> None:
        """Test that delay is capped at max_delay."""
        call_times = []

        @retry_with_backoff(
            max_attempts=4, initial_delay=0.1, max_delay=0.15, exponential_base=2.0
        )
        def failing_func() -> None:
            call_times.append(time.time())
            raise ValueError("fail")

        with pytest.raises(RetryError):
            failing_func()

        # Third delay should be capped at max_delay
        delay3 = call_times[3] - call_times[2]
        assert delay3 < 0.2  # Should be capped at 0.15, not 0.4

    def test_specific_exception_types(self) -> None:
        """Test that only specified exception types are retried."""

        @retry_with_backoff(
            max_attempts=3, initial_delay=0.01, exceptions=(ValueError,)
        )
        def func_with_specific_exception() -> str:
            raise TypeError("not retryable")

        # TypeError should not be retried, should raise immediately
        with pytest.raises(TypeError, match="not retryable"):
            func_with_specific_exception()

    def test_multiple_exception_types(self) -> None:
        """Test retrying multiple exception types."""
        mock_func = Mock(
            side_effect=[ValueError("fail1"), TypeError("fail2"), "success"]
        )
        decorated = retry_with_backoff(
            max_attempts=3, initial_delay=0.01, exceptions=(ValueError, TypeError)
        )(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_with_function_arguments(self) -> None:
        """Test that function arguments are properly passed through."""

        @retry_with_backoff(max_attempts=2, initial_delay=0.01)
        def func_with_args(a: int, b: str, c: int = 3) -> str:
            if a < 2:
                raise ValueError("too small")
            return f"{a}-{b}-{c}"

        # Should fail and retry
        with pytest.raises(RetryError):
            func_with_args(1, "test", c=5)

        # Should succeed on first try
        result = func_with_args(2, "test", c=5)
        assert result == "2-test-5"


class TestRetryWithBackoffAsync:
    """Test suite for async version of retry_with_backoff decorator."""

    async def test_async_successful_first_attempt(self) -> None:
        """Test that async function succeeds on first attempt."""
        call_count = 0

        @retry_with_backoff()
        async def async_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await async_func()

        assert result == "success"
        assert call_count == 1

    async def test_async_successful_after_retries(self) -> None:
        """Test that async function succeeds after retries."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        async def async_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"

        result = await async_func()

        assert result == "success"
        assert call_count == 3

    async def test_async_all_attempts_fail(self) -> None:
        """Test that async function raises RetryError after all attempts."""

        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        async def async_func() -> None:
            raise ValueError("persistent error")

        with pytest.raises(RetryError) as exc_info:
            await async_func()

        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, ValueError)

    async def test_async_exponential_backoff_timing(self) -> None:
        """Test that async delays follow exponential backoff."""
        call_times = []

        @retry_with_backoff(max_attempts=3, initial_delay=0.1, exponential_base=2.0)
        async def async_func() -> None:
            call_times.append(time.time())
            raise ValueError("fail")

        with pytest.raises(RetryError):
            await async_func()

        assert len(call_times) == 3

        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.08 < delay1 < 0.15
        assert 0.18 < delay2 < 0.25

    async def test_async_with_arguments(self) -> None:
        """Test that async function arguments are properly passed."""

        @retry_with_backoff(max_attempts=2, initial_delay=0.01)
        async def async_func(x: int, y: str = "default") -> str:
            if x < 5:
                raise ValueError("too small")
            await asyncio.sleep(0.01)
            return f"{x}-{y}"

        with pytest.raises(RetryError):
            await async_func(1, y="test")

        result = await async_func(5, y="test")
        assert result == "5-test"


class TestRetryError:
    """Test suite for RetryError exception."""

    def test_retry_error_attributes(self) -> None:
        """Test that RetryError stores correct attributes."""
        original_error = ValueError("original")
        retry_error = RetryError(attempts=5, last_exception=original_error)

        assert retry_error.attempts == 5
        assert retry_error.last_exception is original_error
        assert "Failed after 5 attempts" in str(retry_error)
        assert "original" in str(retry_error)

    def test_retry_error_chaining(self) -> None:
        """Test that RetryError properly chains the original exception."""

        @retry_with_backoff(max_attempts=2, initial_delay=0.01)
        def failing_func() -> None:
            raise ValueError("original error")

        with pytest.raises(RetryError) as exc_info:
            failing_func()

        # Check exception chaining
        assert exc_info.value.__cause__ is exc_info.value.last_exception
        assert isinstance(exc_info.value.__cause__, ValueError)
