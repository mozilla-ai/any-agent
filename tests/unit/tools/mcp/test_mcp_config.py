import warnings

import pytest

from any_agent.config import MCPSse


def test_sse_deprecation() -> None:
    sse1 = MCPSse(url="test.example.com:8888/sse")  # noqa: F841
    warnings.filterwarnings("error")
    with pytest.raises(DeprecationWarning):
        sse2 = MCPSse(url="test.example.com:8888/sse")  # noqa: F841
