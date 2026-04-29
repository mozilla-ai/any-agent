"""Re-export tinyagent's span-end callback so identity-based checks line up.

`tinyagent.TinyAgent` registers `tinyagent.callbacks.span_end.SpanEndCallback`
on construction. By aliasing it here, both `any-agent`'s and the inner
TinyAgent's `_add_span_callbacks` see the same class and avoid double-registering.
"""

from tinyagent.callbacks.span_end import SpanEndCallback

__all__ = ["SpanEndCallback"]
