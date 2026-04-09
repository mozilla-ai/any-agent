---
title: Logging
description: Logging setup and customization
---

# Logging with `any-agent`

`any-agent` comes with a logger powered by [Rich](https://github.com/Textualize/rich).

## Quick Start

By default, logging is set up for you. But if you want to customize it, you can call:

```python
from any_agent.logging import setup_logger

setup_logger()
```

## Customizing the Logger

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

## `any_agent.logging.setup_logger()`

Configure the any_agent logger with the specified settings.

```python
def setup_logger(
    level: int = 40,
    rich_tracebacks: bool = True,
    log_format: str | None = None,
    propagate: bool = False,
    **kwargs: Any,
) -> None
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `int` | 40 | The logging level to use (default: logging.INFO) |
| `rich_tracebacks` | `bool` | True | Whether to enable rich tracebacks (default: True) |
| `log_format` | `str \| None` | None | Optional custom log format string |
| `propagate` | `bool` | False | Whether to propagate logs to parent loggers (default: False) |
| `**kwargs` | `Any` | *required* | Additional keyword arguments to pass to RichHandler |
