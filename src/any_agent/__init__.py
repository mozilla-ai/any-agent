from importlib.metadata import PackageNotFoundError, version

from .config import AgentConfig, AgentFramework
from .frameworks.any_agent import AgentRunError, AnyAgent
from .tracing.agent_trace import AgentTrace

try:
    __version__ = version("any-agent")
except PackageNotFoundError:
    # In the case of local development
    # i.e., running directly from the source directory without package being installed
    __version__ = "0.0.0-dev"

__all__ = [
    "AgentConfig",
    "AgentFramework",
    "AgentRunError",
    "AgentTrace",
    "AnyAgent",
    "__version__",
]
