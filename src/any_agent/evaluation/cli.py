import json
import os

from fire import Fire

from any_agent.config import AgentFramework
from any_agent.evaluation import EvaluationCase, evaluate
from any_agent.logging import logger
from any_agent.tracing.trace import AgentSpan, AgentTrace


def do_eval(
    evaluation_case_path: str,
    trace_path: str,
    agent_framework: AgentFramework,
    output_path: str = "output/results.json",
) -> None:
    logger.info("Starting evaluation...")
    logger.info("Loading test case from %s", evaluation_case_path)
    evaluation_case = EvaluationCase.from_yaml(evaluation_case_path)
    logger.info("Loading tracing from %s", trace_path)
    with open(trace_path, encoding="utf-8") as f:
        spans = json.loads(f.read())
    spans = [AgentSpan.model_validate_json(span) for span in spans]
    trace = AgentTrace(spans=spans)
    result = evaluate(
        evaluation_case=evaluation_case,
        trace=trace,
        agent_framework=agent_framework,
    )
    logger.info(f"Final score: {result.score}")

    logger.info("Writing results to %s", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))


def main() -> None:
    Fire(do_eval)  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
