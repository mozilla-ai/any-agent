import pathlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mktestdocs import check_md_file


# Note the use of `str`, makes for pretty output
# Exclude any files that you have custom mocking for.
@pytest.mark.parametrize(
    "fpath",
    [f for f in pathlib.Path("docs").glob("**/*.md") if f.name != "evaluation.md"],
    ids=str,
)
def test_files_all(fpath: pathlib.Path) -> None:
    if fpath.name == "serving.md":
        # the serving markdown runs multiple servers in different processes
        # which is not supported by this testing.
        pytest.skip("Serving.md not supported by docs tester")

    mock_agent = MagicMock()
    mock_create = MagicMock(return_value=mock_agent)
    mock_a2a_tool = AsyncMock()

    mock_create_async = AsyncMock()
    with (
        patch("builtins.open", new_callable=MagicMock),
        patch("any_agent.AnyAgent.create", mock_create),
        patch("any_agent.AnyAgent.create_async", mock_create_async),
        patch("any_agent.tools.a2a_tool_async", mock_a2a_tool),
    ):
        check_md_file(fpath=fpath, memory=True)


def test_evaluation_md() -> None:
    mock_trace = MagicMock()
    mock_trace.tokens.total_tokens = 500
    mock_trace.spans = [1, 2, 3]
    mock_trace.final_output = "Paris"
    mock_trace.spans_to_messages.return_value = []

    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace
    mock_create = MagicMock(return_value=mock_agent)
    mock_a2a_tool = AsyncMock()

    def mock_run_method(*args: Any, **kwargs: Any) -> Any:
        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.reasoning = "Mock evaluation result"
        mock_result.confidence_score = 0.95
        mock_result.suggestions = ["Mock suggestion 1", "Mock suggestion 2"]
        return mock_result

    mock_judge = MagicMock()
    mock_judge.run.side_effect = mock_run_method

    mock_create_async = AsyncMock()
    with (
        patch("builtins.open", new_callable=MagicMock),
        patch("any_agent.AnyAgent.create", mock_create),
        patch("any_agent.AnyAgent.create_async", mock_create_async),
        patch("any_agent.tools.a2a_tool_async", mock_a2a_tool),
        patch("any_agent.evaluation.LlmJudge", return_value=mock_judge),
        patch("any_agent.evaluation.AgentJudge", return_value=mock_judge),
    ):
        check_md_file(fpath=pathlib.Path("docs/evaluation.md"), memory=True)
