import os
from pathlib import Path

import pandas as pd

from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.evaluation.results_saver import save_evaluation_results
from any_agent.tracing.trace import AgentTrace


def test_save_evaluation_results_creates_file(
    evaluation_case: EvaluationCase, agent_trace: AgentTrace, tmp_path: Path
) -> None:
    output_path = str(tmp_path / "test_output.json")

    save_evaluation_results(
        evaluation_case=evaluation_case,
        output_path=output_path,
        output_message="Test evaluation completed.",
        trace=agent_trace,
        hypothesis_answer="This is a test answer.",
        passed_checks=1,
        failed_checks=0,
        score=100.0,
    )

    assert os.path.exists(output_path), "Output file was not created."

    # Clean up after test
    os.remove(output_path)


def test_save_evaluation_results_writes_correct_data(
    evaluation_case: EvaluationCase, agent_trace: AgentTrace, tmp_path: Path
) -> None:
    output_path = str(tmp_path / "test_output.json")
    if os.path.exists(output_path):
        os.remove(output_path)  # Clean up before test

    save_evaluation_results(
        evaluation_case=evaluation_case,
        output_path=output_path,
        output_message="Test evaluation completed.",
        trace=agent_trace,
        hypothesis_answer="This is a test answer.",
        passed_checks=1,
        failed_checks=0,
        score=100.0,
    )

    df = pd.read_json(output_path, orient="records", lines=True)

    assert len(df) == 1, "DataFrame should contain one record."
    assert df.iloc[0]["hypothesis_answer"] == "This is a test answer."
    assert df.iloc[0]["passed_checks"] == 1
    assert df.iloc[0]["failed_checks"] == 0
    assert df.iloc[0]["score"] == 100.0

    # Clean up after test
    os.remove(output_path)
