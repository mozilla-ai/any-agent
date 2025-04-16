import logging

from rich.logging import RichHandler

logger = logging.getLogger("any_agent")
logger.setLevel(logging.DEBUG)
logger.addHandler(RichHandler(rich_tracebacks=True))
