---
title: Serving
description: ServerHandle reference
---

## ServerHandle

A handle for managing an async server instance.

This class provides a clean interface for managing the lifecycle of a server without requiring manual management of the underlying task and server objects.

Lifecycle management for async servers returned by `AnyAgent.serve_async()`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `task` | `asyncio.Task` | The server task |
| `server` | `UvicornServer` | The uvicorn server instance |

### `ServerHandle.shutdown()`

Gracefully shutdown the server with a timeout.

```python
async def shutdown(
    self,
    timeout_seconds: float = 10.0,
) -> None
```

### `ServerHandle.is_running()`

Check if the server is still running.

```python
def is_running(
    self,
) -> bool
```

### Properties

- `port` - `int`: The actual server port (useful when port=0 for OS-assigned ports).
