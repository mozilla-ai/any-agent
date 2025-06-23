import logging
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest


@pytest.fixture
def local_logger(monkeypatch: Any) -> Generator[logging.Logger, None, None]:
    # Create a fresh logger instance for each test
    test_logger = logging.getLogger(f"test_logger_{id(object())}")
    # Remove all handlers
    for handler in test_logger.handlers[:]:
        test_logger.removeHandler(handler)
    test_logger.handlers.clear()
    test_logger.setLevel(logging.NOTSET)
    test_logger.propagate = False
    # Patch the logger in both import styles
    with patch("any_agent.logging.logger", test_logger):
        yield test_logger
