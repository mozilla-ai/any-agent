from unittest.mock import patch, MagicMock

import pytest

from any_agent.tracing import get_tracer_provider, setup_tracing
from any_agent.schema import AgentFramework


def test_get_tracer_provider(tmp_path):
    mock_trace = MagicMock()
    mock_tracer_provider = MagicMock()

    with (
        patch("any_agent.tracing.trace", mock_trace),
        patch("any_agent.tracing.TracerProvider", mock_tracer_provider),
    ):
        get_tracer_provider(
            project_name="test_project",
            output_dir=tmp_path / "telemetry",
            agent_framework=AgentFramework.OPENAI,
        )
        assert (tmp_path / "telemetry").exists()
        mock_trace.set_tracer_provider.assert_called_once_with(
            mock_tracer_provider.return_value
        )


def test_invalid_agent_framework():
    with pytest.raises(NotImplementedError, match="tracing is not supported"):
        setup_tracing(MagicMock(), "invalid_agent_framework")
