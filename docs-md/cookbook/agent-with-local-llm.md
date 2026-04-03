# Local Agent

_(no Google Colab for this one since it is meant to be run entirely locally)_

This tutorial will guide on how to run agent, fully locally / offline i.e., running with a local LLM and a couple of local tools (Callable Python functions), so no data will leave your machine!

This can be especially useful for privacy-sensitive applications or when you want to avoid any cloud dependencies.

In this example, we will showcase how to let an agent read and write in your local filesystem! Specifically,
we will give read-access to the agent to some files in our codebase and ask it to create a short summary of what this code does, and write it to a file.

## Install Dependencies

any-agent uses the python asyncio module to support async functionality. When running in Jupyter notebooks,
 this means we need to enable the use of nested event loops. We'll install any-agent and enable this below using nest_asyncio.

```python
%pip install 'any-agent' --quiet
%pip install ipywidgets --quiet

from pathlib import Path

import nest_asyncio

nest_asyncio.apply()
```

## Set up your own LLM locally

Regardless of which agent framework you choose in any-agent, all of them support any-llm, which is a proxy that
allows us to use whichever LLM inside the framework, hosted on by any provider. For example,
we could use a local model via llama.cpp or [llamafile](https://github.com/Mozilla-Ocho/llamafile), a
google hosted gemini model, or a AWS bedrock hosted Llama model. For this example,
we will use [Ollama](https://ollama.com/) to run our LLM locally!

### Ollama setup

First, install Ollama by following their instructions: https://ollama.com/download

### Picking an LLM

Pick a model that you can run locally based on your hardware and download it from your terminal. For example:

16-24GB RAM -> qwen2.5:14b (Recommended for Agents), granite3.3:8b, or gemma3:12b with ~10–20k context length

24+GB RAM -> `mistral-small3.2:24b` or `devstral:24b` with ~20k+ context length

**_NOTE:_** Smaller models have shown to be more inconsistent and less capable, especially as task complexity increases. If you have the hardware, we highly recommend using the larger models like `mistral-small3.2:24b` for better performance.

### Serving the model with the appropriate context length

By default, Ollama forces a context length of 8192 tokens to all models, which is not enough for our agent to work properly.
We can simply pass to the AgentConfig of `any-agent` as a model argument (`num_ctx` is ollama-specific) our desired value of context length like so:
```
    AgentConfig(
        model_id="ollama/qwen2.5:14b",
        instructions="You must use the available tools to solve the task.",
        tools=[mcp_filesystem, show_plan],
        model_args={"num_ctx": 32000},
    )
```

References: [AgentConfig](../api/config.md), [num_ctx](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)

All four of the models above have a max context length of 128k tokens, but if you have limited RAM if you
set it to 128k it might cause you memory issues. For this example, we will set it to 32,000 tokens and provide a relatively small codebase.

### Load the model

Firstly, download the Ollama model locally:

```
ollama pull qwen2.5:14b
```

>**Note**: You will need [Ollama CLI installed](https://docs.ollama.com/)

## Configure the Agent and the Tools

### Pick which tools to use

Since we want our agent to work fully locally/offline, we will not add any tools that require communication with remote
servers. In this example we are using python callable functions as tools, but we could also have used MCP servers that run fully locally (e.g. [mcp/filesystem](https://hub.docker.com/r/mcp/filesystem))

```python
def read_file(file_path: str) -> str:
    """
    Read a file from the file path given and return its content.

    Args:
        file_path: The path of the file to read.
    """
    try:
        with open(file_path) as file:
            content = file.read()
    except Exception as e:
        content = f"Error reading file: {file_path} \n\n {e}"
    return content
```

```python
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file at the specified file path.

    Args:
        file_path: The path of the file to write.
        content: The content to write to the file.
    """
    try:
        abs_file_path = Path(file_path).resolve()
        with open(abs_file_path, "w") as file:
            file.write(content)
        result = f"Successfully wrote to file: {abs_file_path}"
    except Exception as e:
        result = f"Error writing file: {abs_file_path} \n\n {e}"
    return result
```

Now that you have downloaded your LLM and you have defined your tools, you need to pick your
agent framework to build your agent. Note that the agent you'll build with any-agent can be run across multiple
agent frameworks (Smolagent, TinyAgent, OpenAI, etc) and across various LLMs (Llama, DeepSeek, Mistral, etc).
For this example, we will use the tinyagent framework.

```python
from any_agent import AgentConfig, AnyAgent

model_id = "ollama/qwen2.5:14b"

model_args = {
    "num_ctx": 32000,
    "temperature": 0.0,
    # Note: We use a strict 'WAIT' stop token to force the local model to pause and let the Python kernel execute the tool.
    # Without this, local models often hallucinate the tool output.
    "stop": ["WAIT"],
}

agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id=model_id,
        instructions="""You are a precise tool-calling agent.

        PROTOCOL:
        1. Output valid JSON for the tool you want to use.
        2. AFTER the JSON, write the word "WAIT".
        3. The system will then pause and run the tool for you.
        FORMAT:
        {
            "name": "tool_name",
            "arguments": {
                "arg_name": "value"
            }
        }
        WAIT
        """,
        tools=[read_file, write_file],
        model_args=model_args,
    ),
)
```

## Run the Agent

```python
abs_path = Path("../../demo/app.py").resolve()
output_dir = Path("agent_outputs")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "code_summary.txt"

# --- STEP 1: FORCE READ ---
print("🤖 Step 1: Reading file...")
read_agent = agent.run(f"Output the JSON to read the file at: '{abs_path}'")
# The framework executes the tool and returns the trace
# We assume the last message contains the file content (or we grab it manually)
# Note: In a real app, you'd extract the tool output programmatically.
# For this cookbook, we can just feed the file content explicitly if the agent fails,
# but usually, the agent trace contains the history.

# --- STEP 2: FORCE WRITE ---
# We feed the Previous History + New Instruction
print("🤖 Step 2: Summarizing and Writing...")
write_agent = agent.run(
    f"""
    I have read the file.
    Here is the content: {read_file(str(abs_path))}
    Task: Write a summary of this content to: '{output_file}'
    """
)
```

## View the results

The `agent.run` method returns an AgentTrace object, which has a few convenient attributes for displaying some interesting information about the run.

```python
from any_agent.tracing.attributes import GenAI

def print_step_analysis(step_name, trace):
    print(f"\n=== Analysis: {step_name} ===")
    print(f"Duration: {trace.duration.total_seconds():.2f} seconds")

    tools_found = False
    for span in trace.spans:
        if span.is_tool_execution():
            tool_name = span.attributes.get(GenAI.TOOL_NAME, "Unknown")
            print(f"✓ Tool Called: {tool_name}")
            tools_found = True

    if not tools_found:
        print("⚠️  No tools were called in this step.")
        # Optional: Print raw output to debug why
        print(f"Raw Output: {trace.final_output[:100]}...")

# 1. Inspect the Read Step
print_step_analysis("Step 1 (Force Read)", read_agent)

# 2. Inspect the Write Step
print_step_analysis("Step 2 (Summarize & Write)", write_agent)

# 3. Verify the physical file
print("\n=== Final File Verification ===")
if output_file.exists():
    print(f"✓ SUCCESS: File created at: {output_file.name}")
    print("\n📄 File Contents:")
    print("-" * 50)
    print(output_file.read_text())
    print("-" * 50)
else:
    print(f"✗ FAILURE: File not found at {output_file}")
```
