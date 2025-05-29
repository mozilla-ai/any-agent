import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mktestdocs import check_md_file

from any_agent.evaluation import TraceEvaluationResult


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_files_all(fpath: pathlib.Path) -> None:
    if fpath.name == "serving.md":
        # the serving markdown runs multiple servers in different processes
        # which is not supported by this testing.
        pytest.skip("Serving.md not supported by docs tester")
    mock_agent = MagicMock()
    mock_create = MagicMock(return_value=mock_agent)
    mock_eval = MagicMock()
    mock_eval.return_value = MagicMock(spec=TraceEvaluationResult)
    mock_a2a_tool = AsyncMock()

    mock_create_async = AsyncMock()
    with (
        patch("builtins.open", new_callable=MagicMock),
        patch("any_agent.AnyAgent.create", mock_create),
        patch("any_agent.evaluation.evaluate", mock_eval),
        patch("any_agent.AnyAgent.create_async", mock_create_async),
        patch("any_agent.tools.a2a_tool", mock_a2a_tool),
    ):
        check_md_file(fpath=fpath, memory=True)
