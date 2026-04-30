---
title: Evaluation
description: LlmJudge and AgentJudge reference
---

## LlmJudge

Evaluates agent performance by passing trace text and a question to an LLM.

### Constructor

```python
def __init__(
    self,
    model_id: str,
    framework: AgentFramework = "tinyagent",
    output_type: type[BaseModel] = <class 'schemas.EvaluationOutput'>,
    model_args: dict[str, Any] | None = None,
    system_prompt: str = "You are an expert evaluator that analyzes contextual information to answer specific questions about agent performance and behavior.

You will be provided with:
1. Contextual information of an agent's execution that may be relevant to the evaluation question
2. A specific evaluation question to answer

Your task is to carefully analyze the context and provide a judgment on whether the agent's performance meets the criteria specified in the question.

EVALUATION GUIDELINES:
- Be objective and thorough in your analysis
- If the question asks about specific actions, look for evidence of those actions in the context
- If unsure, err on the side of being more critical rather than lenient

Your output must match the following JSON schema:
{response_schema}",
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `model_id` | `str` | *required* |
| `framework` | `AgentFramework` | "tinyagent" |
| `output_type` | `type[BaseModel]` | <class 'schemas.EvaluationOutput'> |
| `model_args` | `dict[str, Any] \| None` | None |
| `system_prompt` | `str` | "You are an expert evaluator that analyzes contextual information to answer specific questions about agent performance and behavior.

You will be provided with:
1. Contextual information of an agent's execution that may be relevant to the evaluation question
2. A specific evaluation question to answer

Your task is to carefully analyze the context and provide a judgment on whether the agent's performance meets the criteria specified in the question.

EVALUATION GUIDELINES:
- Be objective and thorough in your analysis
- If the question asks about specific actions, look for evidence of those actions in the context
- If unsure, err on the side of being more critical rather than lenient

Your output must match the following JSON schema:
{response_schema}" |

### `LlmJudge.run()`

Run the judge synchronously.

```python
def run(
    self,
    context: str,
    question: str,
    prompt_template: str = "Please answer the evaluation question given the following contextual information:

CONTEXT:
{context}

EVALUATION QUESTION:
{question}",
) -> BaseModel
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `str` | *required* | Any relevant information that may be needed to answer the question |
| `question` | `str` | *required* | The question to ask the agent |
| `prompt_template` | `str` | "Please answer the evaluation question given the following contextual information:

CONTEXT:
{context}

EVALUATION QUESTION:
{question}" | The prompt to use for the LLM |

### `LlmJudge.run_async()`

Run the LLM asynchronously.

```python
async def run_async(
    self,
    context: str,
    question: str,
    prompt_template: str = "Please answer the evaluation question given the following contextual information:

CONTEXT:
{context}

EVALUATION QUESTION:
{question}",
) -> BaseModel
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `str` | *required* | Any relevant information that may be needed to answer the question |
| `question` | `str` | *required* | The question to ask the agent |
| `prompt_template` | `str` | "Please answer the evaluation question given the following contextual information:

CONTEXT:
{context}

EVALUATION QUESTION:
{question}" | The prompt to use for the LLM |

---

## AgentJudge

An agent that evaluates the correctness of another agent's trace.

Agent-based evaluator with built-in tools for trace inspection.

### Constructor

```python
def __init__(
    self,
    model_id: str,
    framework: AgentFramework = "tinyagent",
    output_type: type[BaseModel] = <class 'schemas.EvaluationOutput'>,
    model_args: dict[str, Any] | None = None,
)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `model_id` | `str` | *required* |
| `framework` | `AgentFramework` | "tinyagent" |
| `output_type` | `type[BaseModel]` | <class 'schemas.EvaluationOutput'> |
| `model_args` | `dict[str, Any] \| None` | None |

### `AgentJudge.run()`

Run the agent judge.

```python
def run(
    self,
    trace: AgentTrace,
    question: str,
    additional_tools: list[Callable[[], Any]] | None = None,
) -> AgentTrace
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trace` | `AgentTrace` | *required* | The agent trace to evaluate |
| `question` | `str` | *required* | The question to ask the agent |
| `additional_tools` | `list[Callable[[], Any]] \| None` | None | Additional tools to use for the agent |

### `AgentJudge.run_async()`

Run the agent judge asynchronously.

```python
async def run_async(
    self,
    trace: AgentTrace,
    question: str,
    additional_tools: list[Callable[[], Any]] | None = None,
) -> AgentTrace
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trace` | `AgentTrace` | *required* | The agent trace to evaluate |
| `question` | `str` | *required* | The question to ask the agent |
| `additional_tools` | `list[Callable[[], Any]] \| None` | None | Additional tools to use for the agent |
