import pathlib
from unittest.mock import MagicMock, patch

import pytest
from mktestdocs import check_md_file


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_files_all(fpath: pathlib.Path) -> None:
    # Create a mock for evaluate_telemetry
    mock_evaluate = MagicMock()
    mock_agent = MagicMock()
    mock_create = MagicMock(return_value=mock_agent)
    # Patch the evaluate_telemetry function.
    # Eventually we may want to better validate that the docs use evaluate_telemetry correctly
    with (
        patch("any_agent.evaluation.evaluate.evaluate_telemetry", mock_evaluate),
        patch("any_agent.AnyAgent.create", mock_create),
    ):
        check_md_file(fpath=fpath, memory=True)  # type: ignore[no-untyped-call]
