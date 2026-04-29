"""Re-export the Callback base class from `tinyagent` so callback identities
are unified across `any-agent` and `tinyagent`.
"""

from tinyagent.callbacks.base import Callback

__all__ = ["Callback"]
