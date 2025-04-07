# Agent Evaluation

!!! warning

    The codebase for evaluation is under development and is not yet stable. Use with caution,
    we welcome contributions.

Evaluation using any_agent.evaluation is designed to be a "trace-first" evaluation. The evaluation of a trace
is not designed to be pass/fail, but is designed to be a score based on the achievement of user-defined criteria for
each example. Agent systems are hyper-specific to each use case, and it's difficult to provide a single set of metrics
that would reliably provide the insight needed to make a decision about the effectiveness of an agent.

Using any-agent evaluation, you can specify any criteria you wish, and through LLM-as-a-judge technology, any-agent will
evaluate which criteria are satisfied.

## Example

Using the unified tracing format provided by any-agent's [tracing functionality](./tracing.md), the trace can be evaluated
with user defined criteria. The steps for evaluating an agent are as follows:

1. Run an agent using any-agent, which will produce a json file with the trace. For example

```python
from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tracing import setup_tracing

main_agent = AgentConfig(
	model_id="gpt-4o-mini",
    tools=["any_agent.tools.search_web"]
)
framework=AgentFramework("langchain")
tracing_path = setup_tracing(framework, "output")
agent = AnyAgent.create(framework, main_agent)

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
```
1. Define a test case in a yaml file, e.g.

```yaml
# The criteria will be passed to an llm-as-a-judge along with the trace to have as context
# The points specify the weight given to each criteria when producing the final score
llm_judge: openai/gpt-4o
checkpoints:
  - points: 1
    criteria: Ensure that the agent called the search_web tool in order to retrieve the length of Pont des Arts
  - points: 1
    criteria: Ensure that the agent called the search_web tool in order to access the top speed of a leopard
  - points: 1
    criteria: |
        Ensure that the agent ran a python snippet to combine the information
        from the info retrieved from the web searches

# Optionally, you can check whether the final answer is what was expected. Checking this value does not use an LLM
ground_truth:
  - name: Time
    points: 5
    value: 9.63
```

1. Run the evaluation using the test case and trace.
```python
from any_agent.evaluation.test_case import TestCase
from any_agent.evaluation.evaluate import evaluate_telemetry
test_case = TestCase.from_yaml("/path/to/test/case/yaml")
evaluate_telemetry(test_case, '/path/to/telemetry/output')
```
The output will look something like this:

```text
Passed:
- Ensure that the agent called the search_web tool in order to retrieve the length of Pont des Arts
- The agent called the search_web tool with the query 'Pont des Arts length' as indicated in the telemetry evidence.

Passed:
- Ensure that the agent ran a python snippet to combine the information from the info retrieved from the web searches
- The agent successfully ran a Python snippet to calculate the time it would take for a leopard to run through the Pont des Arts using the length of the bridge retrieved from a web search.

Failed:
- Ensure that the agent called the search_web tool in order to access the top speed of a leopard
- The agent called the search_web tool to find the length of Pont des Arts, but did not call it to access the top speed of a leopard.

Failed:
- Check if Time is approximately '9.63'.
- The calculated time in the agent's answer is 9.62, not 9.63.

Failed:
- Is the answer a direct match?
- Partial Match (F1) score is 0.0
Passed checkpoints: 2
Failed checkpoints: 3
=====================================
Score: 2/9
=====================================

Reading existing output from output/results.json
Writing output to output/results.json
```
