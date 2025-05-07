# Agent Evaluation

!!! warning

    The codebase for evaluation is under development and is not yet stable. Use with caution,
    we welcome contributions.

Evaluation using any_agent.evaluation is designed to be a "trace-first" evaluation. The evaluation of a trace
is not designed to be pass/fail, but is rather a score based on the achievement of user-defined criteria for
each example. Agent systems are hyper-specific to each use case, and it's difficult to provide a single set of metrics
that would reliably provide the insight needed to make a decision about the effectiveness of an agent.

Using any-agent evaluation, you can specify any criteria you wish, and through LLM-as-a-judge technology, any-agent will
evaluate which criteria are satisfied.

## Example

Using the unified tracing format provided by any-agent's [tracing functionality](./tracing.md), the trace can be evaluated
with user defined criteria. The steps for evaluating an agent are as follows:

### Run an agent using any-agent, which will produce a trace. For example

```python
from any_agent import AgentConfig, AnyAgent, TracingConfig
from any_agent.tools import search_web

agent = AnyAgent.create(
    "langchain",
    AgentConfig(
        model_id="gpt-4o-mini",
        tools=[search_web]
    ),
    tracing=TracingConfig(console=True, cost_info=False)
)

agent_trace = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

```


### Define an evaluation case either in a yaml file or in python:

=== "YAML"
    ~~~yaml
    {% include "./examples/evaluation_case.yaml" %}
    ~~~
    Then in python
    ```python
    from any_agent.evaluation.evaluation_case import EvaluationCase
    evaluation_case = EvaluationCase.from_yaml(evaluation_case_path)
    ```

=== "Python"
    ```python
    from any_agent.evaluation.evaluation_case import EvaluationCase
    evaluation_case = EvaluationCase(
            ground_truth=[{"name": "Test Case 1", "value": 1.0, "points": 1.0}],
            checkpoints=[{"criteria": "Check if value is 1.0", "points": 1}],
            llm_judge="gpt-4o-mini",
            final_output_criteria=[]
    )
    ```

### Run the evaluation using the test case and trace.

```python
from any_agent.evaluation import EvaluationRunner
from any_agent.evaluation.evaluation_case import EvaluationCase
output_path="tmp/path/result.json"
evaluation_case = EvaluationCase(
    ground_truth=[{"name": "Test Case 1", "value": 1.0, "points": 1.0}],
    checkpoints=[{"criteria": "Check if value is 1.0", "points": 1}],
    llm_judge="gpt-4o-mini",
    final_output_criteria=[]
)
runner = EvaluationRunner(output_path=output_path)
runner.add_evaluation_case(evaluation_case)
runner.add_trace(agent_trace, 'OPENAI')
runner.run()
```
The output will look something like this:

```text
Passed:
- Ensure that the agent called the search_web tool in order to retrieve the length of Pont des Arts
- The agent called the search_web tool with the query 'Pont des Arts length' as indicated in the trace evidence.

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


## Command Line

If you have the file and test case prepared, a command line tools is provided for convenience called `any-agent-evaluate`.

It can be called like so

```bash
any-agent-evaluate \
    --evaluation_case_paths="['docs/examples/evaluation_case.yaml']" \
    --trace_paths "['tests/unit/evaluation/sample_traces/OPENAI.json']" \
    --agent_framework 'OPENAI'
```
