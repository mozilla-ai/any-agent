import os
import pathlib
import subprocess

import pytest


@pytest.mark.parametrize(
    "notebook_path",
    list(pathlib.Path("docs/cookbook").glob("*.ipynb")),
    ids=lambda x: x.stem,
)
@pytest.mark.timeout(180)
def test_cookbook_notebook(
    notebook_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Test that cookbook notebooks execute without errors using jupyter execute."""
    if notebook_path.stem == "mcp_agent":
        pytest.skip("See https://github.com/mozilla-ai/any-agent/issues/706")
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "jupyter",
                "execute",
                notebook_path.name,
                "--allow-errors",
                "--output",
                f"executed_{notebook_path.name}",
            ],
            cwd="docs/cookbook",  # Run in cookbook directory like original action
            env={
                "MISTRAL_API_KEY": os.environ["MISTRAL_API_KEY"],
                "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
                "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                "HF_TOKEN": os.environ["HF_TOKEN"],
                "HF_ENDPOINT": os.environ["HF_ENDPOINT"],
                "PATH": os.environ["PATH"],
                "IN_PYTEST": "1",  # Signal local_llm and mcp_agent notebooks that we are running an automated test.
            },
            timeout=170,  # Time out slightly earlier so that we can log the output.
            capture_output=True,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        # Handle timeout case - log stdout/stderr that were captured before timeout
        stdout = e.stdout.decode() if e.stdout else "(no stdout captured)"
        stderr = e.stderr.decode() if e.stderr else "(no stderr captured)"
        pytest.fail(
            f"Notebook {notebook_path.name} timed out after 2 minutes\n stdout: {stdout}\n stderr: {stderr}"
        )

    if result.returncode != 0:
        stdout = result.stdout.decode() if result.stdout else "(no stdout captured)"
        stderr = result.stderr.decode() if result.stderr else "(no stderr captured)"
        pytest.fail(
            f"Notebook {notebook_path.name} failed with return code {result.returncode}\n stdout: {stdout}\n stderr: {stderr}"
        )
