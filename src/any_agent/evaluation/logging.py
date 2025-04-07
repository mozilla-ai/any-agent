import sys
from loguru import logger

# Remove default logger
logger.remove()

# Add custom colored logger
logger = logger.opt(ansi=True)
logger.add(sys.stdout, colorize=True, format="{message}")


# Export configured logger
def get_logger():
    return logger
