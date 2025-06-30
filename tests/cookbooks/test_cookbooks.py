import os
import pathlib
import subprocess

import pytest


@pytest.mark.parametrize(
    "notebook_path",
    list(pathlib.Path("docs/cookbook").glob("*.ipynb")),
    ids=lambda x: x.stem,
)
def test_cookbook_notebook(
    notebook_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Test that cookbook notebooks execute without errors using jupyter execute."""
    try:
        result = subprocess.run(
            ["jupyter", "execute", notebook_path.name],
            cwd="docs/cookbook",  # Run in cookbook directory like original action
            env={
                "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                "PATH": os.environ["PATH"],
                "IN_PYTEST": "1",  # For mcp_agent notebook which needs to mock input
            },
            timeout=29, # one second less than the timeout in pytest
            capture_output=True,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        # Handle timeout case - log stdout/stderr that were captured before timeout
        stdout = e.stdout.decode() if e.stdout else "(no stdout captured)"
        stderr = e.stderr.decode() if e.stderr else "(no stderr captured)"
        pytest.fail(f"Notebook {notebook_path.name} timed out after 29 seconds\n stdout: {stdout}\n stderr: {stderr}")
    
    if result.returncode != 0:
        stdout = result.stdout.decode() if result.stdout else "(no stdout captured)"
        stderr = result.stderr.decode() if result.stderr else "(no stderr captured)"
        pytest.fail(f"Notebook {notebook_path.name} failed with return code {result.returncode}\n stdout: {stdout}\n stderr: {stderr}")