from unittest.mock import patch, MagicMock

import pytest

from any_agent.tracing import _get_tracer_provider, setup_tracing
from any_agent.config import AgentFramework, TracingConfig


def test_get_tracer_provider(tmp_path):
    mock_trace = MagicMock()
    mock_tracer_provider = MagicMock()

    with (
        patch("any_agent.tracing.trace", mock_trace),
        patch("any_agent.tracing.TracerProvider", mock_tracer_provider),
    ):
        _get_tracer_provider(
            output_dir=tmp_path / "traces",
            agent_framework=AgentFramework.OPENAI,
            tracing_config=TracingConfig(),
        )
        assert (tmp_path / "traces").exists()
        mock_trace.set_tracer_provider.assert_called_once_with(
            mock_tracer_provider.return_value
        )


def test_invalid_agent_framework(tmp_path):
    with pytest.raises(ValueError, match="Unsupported agent type"):
        setup_tracing(MagicMock(), tmp_path / "traces")
