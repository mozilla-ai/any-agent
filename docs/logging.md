# Logging with `any-agent`

 `any-agent` comes with a logger powered by [Rich](https://github.com/Textualize/rich)

## Quick Start

By default, logging is set up for you. But if you want to customize it, you can call:

```python
from any_agent.logging import setup_logger

setup_logger()
```

## Customizing the Logger

The `setup_logger` function lets you tweak how logs look and behave. Here are the options you can play with:

| Argument         | Type      | Default           | Description                                                                 |
|------------------|-----------|-------------------|-----------------------------------------------------------------------------|
| `level`          | int       | `logging.ERROR`   | The logging level (e.g., `logging.INFO`, `logging.DEBUG`).                  |
| `rich_tracebacks`| bool      | `True`            | Show beautiful tracebacks with Rich.                                        |
| `log_format`     | str/None  | `None`            | Custom log format string (uses Python's `logging.Formatter`).               |
| `propagate`      | bool      | `False`           | If `True`, logs also go to parent loggers.                                  |
| `**kwargs`       | Any       |                   | Extra options passed to `RichHandler` (see [Rich docs](https://rich.readthedocs.io/en/stable/logging.html)). |

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

