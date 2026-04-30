---
title: Tools
description: Built-in tools provided by any-agent
---

Built-in callable tools that can be passed directly to `AgentConfig.tools`.

## `any_agent.tools.a2a_tool()`

Perform a query using A2A to another agent (synchronous version).

```python
def a2a_tool(
    url: str,
    toolname: str | None = None,
    http_kwargs: dict[str, Any] | None = None,
) -> Callable[str, str | None, str | None, str]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | *required* | The url in which the A2A agent is located. |
| `toolname` | `str \| None` | None | The name for the created tool. Defaults to `call_{agent name in card}`. Leading and trailing whitespace are removed. Whitespace in the middle is replaced by `_`. |
| `http_kwargs` | `dict[str, Any] \| None` | None | Additional kwargs to pass to the httpx client. |

**Returns:** A sync `Callable` that takes a query and returns the agent response.

## `any_agent.tools.a2a_tool_async()`

Perform a query using A2A to another agent.

```python
async def a2a_tool_async(
    url: str,
    toolname: str | None = None,
    http_kwargs: dict[str, Any] | None = None,
) -> Callable[str, str | None, str | None, Coroutine[Any, Any, dict[str, Any]]]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | *required* | The url in which the A2A agent is located. |
| `toolname` | `str \| None` | None | The name for the created tool. Defaults to `call_{agent name in card}`. Leading and trailing whitespace are removed. Whitespace in the middle is replaced by `_`. |
| `http_kwargs` | `dict[str, Any] \| None` | None | Additional kwargs to pass to the httpx client. |

**Returns:** An async `Callable` that takes a query and returns the agent response.

## `any_agent.tools.ask_user_verification()`

Asks user to verify the given `query`.

```python
def ask_user_verification(
    query: str,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | The question that requires verification. |

## `any_agent.tools.prepare_final_output()`

Prepare instructions and tools for structured output, returning the function directly.

```python
def prepare_final_output(
    output_type: type[BaseModel],
    instructions: str | None = None,
) -> tuple[str, Callable[str, dict[str, str | bool | dict[str, Any] | list[Any]]]]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_type` | `type[BaseModel]` | *required* | The Pydantic model type for structured output |
| `instructions` | `str \| None` | None | Original instructions to modify |

**Returns:** Tuple of (modified_instructions, final_output_function)

## `any_agent.tools.search_tavily()`

Perform a Tavily web search based on your query and return the top search results.

See https://blog.tavily.com/getting-started-with-the-tavily-search-api for more information.

```python
def search_tavily(
    query: str,
    include_images: bool = False,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | The search query to perform. |
| `include_images` | `bool` | False | Whether to include images in the results. |

**Returns:** The top search results as a formatted string.

## `any_agent.tools.search_web()`

Perform a duckduckgo web search based on your query (think a Google search) then returns the top search results.

```python
def search_web(
    query: str,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | The search query to perform. |

**Returns:** The top search results.

## `any_agent.tools.send_console_message()`

Send the specified user a message via console and returns their response.

```python
def send_console_message(
    user: str,
    query: str,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user` | `str` | *required* | The user to ask the question to. |
| `query` | `str` | *required* | The question to ask the user. |

**Returns:** str: The user's response.

## `any_agent.tools.show_final_output()`

Show the final answer to the user.

```python
def show_final_output(
    answer: str,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `answer` | `str` | *required* | The final answer. |

## `any_agent.tools.show_plan()`

Show the current plan to the user.

```python
def show_plan(
    plan: str,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plan` | `str` | *required* | The current plan. |

## `any_agent.tools.visit_webpage()`

Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages.

```python
def visit_webpage(
    url: str,
    timeout: int = 30,
    max_length: int = 10000,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | *required* | The url of the webpage to visit. |
| `timeout` | `int` | 30 | The timeout in seconds for the request. |
| `max_length` | `int` | 10000 | The maximum number of characters of text that can be returned (default=10000). If max_length==-1, text is not truncated and the full webpage is returned. |
