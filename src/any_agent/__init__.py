from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("any-agent")
except PackageNotFoundError:
    __version__ = "unknown"
from .config import AgentConfig, AgentFramework
from .frameworks.any_agent import AgentRunError, AnyAgent
from .tracing.agent_trace import AgentTrace

__all__ = [
    "AgentConfig",
    "AgentFramework",
    "AgentRunError",
    "AgentTrace",
    "AnyAgent",
    "__version__",
]
