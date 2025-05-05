from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from any_agent.config import AgentFramework, TracingConfig
from any_agent.tracing import Tracer


def test_tracer_initialization(tmp_path: Path) -> None:
    mock_trace = MagicMock()
    mock_tracer_provider = MagicMock()

    with (
        patch("any_agent.tracing.trace", mock_trace),
        patch("any_agent.tracing.TracerProvider", mock_tracer_provider),
    ):
        tracer = Tracer(
            agent_framework=AgentFramework.OPENAI,
            tracing_config=TracingConfig(
                output_dir=str(tmp_path / "traces"),
            ),
        )

        # Verify tracer was initialized correctly
        assert tracer.agent_framework == AgentFramework.OPENAI
        assert tracer.tracing_config.output_dir == str(tmp_path / "traces")
        assert tracer.is_enabled is True
        assert (tmp_path / "traces").exists()

        # Verify tracing was set up
        mock_trace.set_tracer_provider.assert_called_once_with(
            mock_tracer_provider.return_value,
        )


def test_tracer_with_unsupported_framework(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="AGNO tracing is not supported."):
        Tracer(
            agent_framework=AgentFramework.AGNO,
            tracing_config=TracingConfig(
                output_dir=str(tmp_path / "traces"), console=False
            ),
        )
