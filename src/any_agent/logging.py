import logging

from rich.logging import RichHandler

logger = logging.getLogger("any_agent")


def setup_logger(
    level: int = logging.ERROR,
    rich_tracebacks: bool = True,
    log_format: str | None = None,
    propagate: bool = False,
    show_time: bool = True,
    show_path: bool = True,
    show_level: bool = True,
) -> None:
    """Configure the any_agent logger with the specified settings.

    Args:
        level: The logging level to use (default: logging.INFO)
        rich_tracebacks: Whether to enable rich tracebacks (default: True)
        log_format: Optional custom log format string
        propagate: Whether to propagate logs to parent loggers (default: False)
        stream: The stream to write logs to (default: sys.stderr)
        show_time: Whether to show timestamps in logs (default: True)
        show_path: Whether to show file paths in logs (default: True)
        show_level: Whether to show log levels in logs (default: True)
        show_locals: Whether to show local variables in tracebacks (default: False)

    """
    logger.setLevel(level)
    logger.propagate = propagate

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Configure RichHandler with specified settings
    handler = RichHandler(
        rich_tracebacks=rich_tracebacks,
        markup=True,
        show_time=show_time,
        show_path=show_path,
        show_level=show_level,
    )

    if log_format:
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

    logger.addHandler(handler)


# Set default configuration
setup_logger()
