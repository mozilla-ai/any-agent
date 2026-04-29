"""Re-export the Callback base class from `tinyagent`.

Re-exporting here keeps callback class identities unified across
`any-agent` and `tinyagent` so isinstance checks behave correctly.
"""

from tinyagent.callbacks.base import Callback

__all__ = ["Callback"]
