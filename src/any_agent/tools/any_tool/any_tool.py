import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

from any_agent.config import AgentFramework, Tool

T = TypeVar("T", bound=Callable[..., Any])


@runtime_checkable
class HasName(Protocol):
    """Protocol for objects that have a name."""

    name: str


class AnyToolBase(ABC, BaseModel, Generic[T]):
    """Abstract base class for wrapping tools for specific frameworks."""

    framework: AgentFramework
    tool: T | Tool

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __eq__(self, other: object) -> bool:
        """Tools are equal if the name is equal."""
        if not isinstance(other, HasName):
            return False

        return self.name == other.name

    @property
    def name(self) -> str:
        """Name of the tool."""
        return (
            self.tool.name
            if isinstance(self.tool, HasName)
            else self.__class__.__name__
        )

    def __getattr__(self, name: str) -> Any:
        """Use AnyTool as proxy of the underlying tool."""
        return getattr(self.tool, name)

    @field_validator("tool", mode="before")
    @classmethod
    @abstractmethod
    def _validate_tool_type(cls, tool: Any) -> T: ...

    @field_validator("tool", mode="before")
    @classmethod
    def _verify_tool(cls, tool: Any) -> Any:
        if not tool.__doc__:
            msg = f"Tool {tool} needs to have a docstring but does not"
            raise ValueError(msg)

        signature = inspect.signature(tool)
        if signature.return_annotation is inspect.Signature.empty:
            msg = f"Tool {tool} needs to have a return type but does not"
            raise ValueError(msg)

        signature = inspect.signature(tool)
        for param in signature.parameters.values():
            if param.annotation is inspect.Signature.empty:
                msg = f"Tool {tool} needs to have typed arguments but does not"
                raise ValueError(msg)

        return tool