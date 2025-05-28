# Logging with `any-agent`

 `any-agent` comes with a logger powered by [Rich](https://github.com/Textualize/rich)

## Quick Start

By default, logging is set up for you. But if you want to customize it, you can call:

```python
from any_agent.logging import setup_logger

setup_logger()
```

## Customizing the Logger

View the docstring in [`setup_logger`][any_agent.logging.setup_logger] for a description of the arguments available .

### Example: Set Log Level to DEBUG

```python
from any_agent.logging import setup_logger
import logging

setup_logger(level=logging.DEBUG)
```

### Example: Custom Log Format

```python
setup_logger(log_format="%(asctime)s - %(levelname)s - %(message)s")
```

### Example: Propagate Logs

```python
setup_logger(propagate=True)
```

::: any_agent.logging.setup_logger
