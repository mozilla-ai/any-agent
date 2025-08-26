# Reasoning and Thinking Tokens Support

any-agent provides comprehensive support for tracking reasoning/thinking tokens from various AI models that expose their internal reasoning process.

## Supported Models and Token Types

### OpenAI o1 Models
- **Reasoning tokens**: Available in `usage.completion_tokens_details.reasoning_tokens`
- **Cached tokens**: When using prompt caching

### Google Gemini 2.5 Models  
- **Thinking tokens**: Available in `usage_metadata.thoughts_token_count`
- Controlled via `thinking_budget` parameter

### DeepSeek Reasoning Models
- **Reasoning content**: Provided separately in `reasoning_content` field
- Requires manual token counting of reasoning content
- Note: reasoning_content should be removed before next API call

## Trace Attributes

The following attributes are added to LLM call spans when available:

- `gen_ai.usage.reasoning_tokens`: Number of tokens used for internal reasoning
- `gen_ai.usage.cached_tokens`: Number of cached prompt tokens (OpenAI)

## Implementation Details

### Token Extraction
- Tokens are only added to spans when > 0
- Graceful handling of missing attributes
- Framework-specific extraction logic

### Backward Compatibility
- No changes to existing token attributes
- Optional parameters maintain compatibility
- Zero values are not added to spans

## Usage Example

When using models with reasoning capabilities, the tokens automatically appear in traces:

```python
# Using OpenAI o1
agent = AnyAgent.create(
    AgentFramework.LANGCHAIN,
    AgentConfig(model_id="openai/o1-preview")
)
trace = agent.run("Complex reasoning task")

# Reasoning tokens will be in trace spans
for span in trace.spans:
    if span.attributes.get("gen_ai.operation_name") == "call_llm":
        reasoning = span.attributes.get("gen_ai.usage.reasoning_tokens", 0)
        print(f"Reasoning tokens: {reasoning}")
```

## Future Enhancements

- Support for DeepSeek reasoning_content token counting
- Reasoning efficiency metrics (output quality / reasoning tokens)
- Latency metrics (TTFT, TPOT) with proper timing implementation
- Multi-turn reasoning analysis