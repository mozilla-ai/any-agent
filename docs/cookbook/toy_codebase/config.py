from typing import Annotated

from pydantic import BaseModel, DirectoryPath, FilePath
from pydantic.functional_validators import AfterValidator

FIND_PROMPT = """
You are given two pieces of information:
1. A list of valid section names.
2. A user question.

Your task is to:
- Identify exactly one `section_name` from the provided list that seems related to the user question.
- Return the `section_name` exactly as it appears in the list.
- Do NOT answer the question.
- Do NOT return any additional text, explanation, or formatting.
- Do NOT combine multiple section names into a single response.

Here is the list of valid section names:

```
{SECTIONS}
```

Now, based on the following question, return the single most relevant `section_name` from the list.
"""

ANSWER_PROMPT = """
You are a rigorous assistant answering questions.
You must only answer based on the current information available which is:

```
{CURRENT_INFO}
```

If the current information available not enough to answer the question,
you must return "I need more info" and nothing else.
"""


def validate_model(value):
    parts = value.split("/")
    if len(parts) != 3:
        raise ValueError("model must be formatted as `owner/repo/file`")
    if not value.endswith(".gguf"):
        raise ValueError("model must be a gguf file")
    return value


def validate_find_prompt(value):
    if "{SECTIONS}" not in value:
        raise ValueError("find_prompt must contain `{SECTIONS}` placeholder")
    return value


def answer_prompt(value):
    if "{CURRENT_INFO}" not in value:
        raise ValueError("answer_prompt must contain `{CURRENT_INFO}` placeholder")
    return value


class Config(BaseModel):
    question: str
    input_file: FilePath
    output_dir: DirectoryPath
    model: Annotated[str, AfterValidator(validate_model)]
    answer_prompt: Annotated[str, AfterValidator(answer_prompt)]
    find_prompt: Annotated[str, AfterValidator(validate_find_prompt)]
