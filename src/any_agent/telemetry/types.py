from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    """String-based enum for span kind to make it serializable."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"

    @classmethod
    def from_otel(cls, kind: Any | None) -> "SpanKind":
        """Convert from OpenTelemetry SpanKind."""
        if kind is None:
            return cls.INTERNAL

        mapping = {
            0: cls.INTERNAL,
            1: cls.SERVER,
            2: cls.CLIENT,
            3: cls.PRODUCER,
            4: cls.CONSUMER,
        }
        return mapping.get(kind.value, cls.INTERNAL)


class TraceFlags(BaseModel):
    """Serializable trace flags."""

    value: int = 0

    @classmethod
    def from_otel(cls, flags: Any | None) -> "TraceFlags":
        """Convert from OpenTelemetry TraceFlags."""
        if flags is None:
            return cls(value=0)
        return cls(value=flags.value if hasattr(flags, "value") else 0)


class TraceState(BaseModel):
    """Serializable trace state."""

    entries: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_otel(cls, state: Any | None) -> "TraceState":
        """Convert from OpenTelemetry TraceState."""
        if state is None:
            return cls()
        return cls(entries=dict(state.items()) if hasattr(state, "items") else {})


class SpanContext(BaseModel):
    """Serializable span context."""

    trace_id: int | None = None
    span_id: int | None = None
    is_remote: bool = False
    trace_flags: TraceFlags = Field(default_factory=TraceFlags)
    trace_state: TraceState = Field(default_factory=TraceState)

    @classmethod
    def from_otel(cls, context: Any | None) -> "SpanContext":
        """Convert from OpenTelemetry SpanContext."""
        if context is None:
            return cls()

        return cls(
            trace_id=getattr(context, "trace_id", None),
            span_id=getattr(context, "span_id", None),
            is_remote=getattr(context, "is_remote", False),
            trace_flags=TraceFlags.from_otel(getattr(context, "trace_flags", None)),
            trace_state=TraceState.from_otel(getattr(context, "trace_state", None)),
        )


class StatusCode(str, Enum):
    """String-based enum for status code to make it serializable."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"

    @classmethod
    def from_otel(cls, code: Any | None) -> "StatusCode":
        """Convert from OpenTelemetry StatusCode."""
        if code is None:
            return cls.UNSET

        mapping = {"UNSET": cls.UNSET, "OK": cls.OK, "ERROR": cls.ERROR}

        if hasattr(code, "name"):
            return mapping.get(code.name, cls.UNSET)
        return cls.UNSET


class Status(BaseModel):
    """Serializable status."""

    status_code: StatusCode = StatusCode.UNSET
    description: str | None = None

    @classmethod
    def from_otel(cls, status: Any | None) -> "Status":
        """Convert from OpenTelemetry Status."""
        if status is None:
            return cls()

        return cls(
            status_code=StatusCode.from_otel(getattr(status, "status_code", None)),
            description=getattr(status, "description", ""),
        )


class AttributeValue(BaseModel):
    """A wrapper for attribute values that can be serialized."""

    value: str | int | float | bool | list[str | int | float | bool]


class Link(BaseModel):
    """Serializable link."""

    context: SpanContext
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_otel(cls, link: Any | None) -> "Link":
        """Convert from OpenTelemetry Link."""
        if link is None:
            return cls(context=SpanContext())

        return cls(
            context=SpanContext.from_otel(getattr(link, "context", None)),
            attributes=getattr(link, "attributes", None),
        )


class Event(BaseModel):
    """Serializable event."""

    name: str
    timestamp: int = 0
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_otel(cls, event: Any | None) -> "Event":
        """Convert from OpenTelemetry Event."""
        if event is None:
            return cls(name="")

        return cls(
            name=getattr(event, "name", ""),
            timestamp=getattr(event, "timestamp", 0),
            attributes=getattr(event, "attributes", None),
        )


class Resource(BaseModel):
    """Serializable resource."""

    attributes: dict[str, Any] = Field(default_factory=dict)
    schema_url: str = ""

    @classmethod
    def from_otel(cls, resource: Any | None) -> "Resource":
        """Convert from OpenTelemetry Resource."""
        if resource is None:
            return cls()

        return cls(
            attributes=getattr(resource, "attributes", {}),
            schema_url=getattr(resource, "schema_url", ""),
        )
