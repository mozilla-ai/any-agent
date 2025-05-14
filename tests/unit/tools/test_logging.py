import logging

from any_agent.logging import setup_logger


def test_setup_logger_sets_level_and_propagate(local_logger: logging.Logger) -> None:
    setup_logger(level=logging.INFO, propagate=False)
    assert local_logger.level == logging.INFO
    assert local_logger.propagate is False
    setup_logger(level=logging.DEBUG, propagate=True)
    assert local_logger.level == logging.DEBUG
    assert local_logger.propagate is True


def test_setup_logger_removes_existing_handlers(local_logger: logging.Logger) -> None:
    dummy_handler = logging.StreamHandler()
    local_logger.addHandler(dummy_handler)
    assert dummy_handler in local_logger.handlers
    setup_logger()
    assert dummy_handler not in local_logger.handlers


def test_setup_logger_with_custom_format(local_logger: logging.Logger) -> None:
    custom_format = "%(levelname)s: %(message)s"
    setup_logger(log_format=custom_format)
    handler = local_logger.handlers[0]
    assert isinstance(handler.formatter, logging.Formatter)
    assert handler.formatter._fmt == custom_format


def test_setup_logger_multiple_calls_idempotent(local_logger: logging.Logger) -> None:
    setup_logger(level=logging.INFO)
    first_handler = local_logger.handlers[0]
    setup_logger(level=logging.WARNING)
    second_handler = local_logger.handlers[0]
    assert first_handler is not second_handler
    assert local_logger.level == logging.WARNING


def test_setup_logger_edits_global_logger(local_logger: logging.Logger) -> None:
    # This test now just checks that setup_logger edits the patched logger
    dummy_handler = logging.StreamHandler()
    local_logger.addHandler(dummy_handler)
    assert dummy_handler in local_logger.handlers
    setup_logger()
    assert dummy_handler not in local_logger.handlers
    assert local_logger.level == logging.ERROR
    assert local_logger.propagate is False
