import pytest

# Skip entire module if a2a dependencies are not available
pytest.importorskip("a2a")


@pytest.mark.asyncio
async def test_a2a_imports() -> None:
    """Test that A2A serving can be imported."""
    from any_agent.serving import serve_a2a_async
