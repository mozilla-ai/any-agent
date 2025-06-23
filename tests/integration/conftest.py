import hashlib
import socket

import pytest

from any_agent.config import AgentFramework


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Add custom command-line options to pytest.

    This hook adds the `--update-trace-assets` flag to pytest, which can be used when running integration tests.
    When this flag is set, tests that generate trace asset files (aka the integration test that
    produces agent traces) will update the asset files in the assets directory.
    This is useful when the expected trace output changes and you
    want to regenerate the reference files.
    """
    parser.addoption(
        "--update-trace-assets",
        action="store_true",
        default=False,
        help="Update trace asset files instead of asserting equality.",
    )


def _is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available for binding.

    This isn't a perfect check but it at least tells us if there is absolutely no chance of binding to the port.

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def _get_deterministic_port(test_name: str, framework_name: str) -> int:
    """Generate a deterministic port number based on test name and framework.

    This ensures each test gets a unique port that remains consistent across runs.
    """
    # Create a unique string by combining test name and framework
    unique_string = f"{test_name}_{framework_name}"

    # Generate a hash and convert to a port number in the range 6000-9999
    hash_value = int(hashlib.md5(unique_string.encode()).hexdigest()[:4], 16)  # noqa: S324
    return 6000 + (hash_value % 4000)


@pytest.fixture
def test_port(request: pytest.FixtureRequest, agent_framework: AgentFramework) -> int:
    """Single fixture that provides a unique, deterministic port for each test."""
    test_name = request.node.name
    framework_name = agent_framework.value

    port = _get_deterministic_port(test_name, framework_name)

    # Ensure the port is available, if not, try nearby ports
    original_port = port
    attempts = 0
    while not _is_port_available(port) and attempts < 50:
        port = original_port + attempts + 1
        attempts += 1

    if not _is_port_available(port):
        msg = f"Could not find an available port starting from {original_port}"
        raise RuntimeError(msg)

    return port
