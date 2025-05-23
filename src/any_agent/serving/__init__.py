import sys

try:
    from .server import _get_a2a_server, serve_a2a
except ImportError as e:
    msg = "You need to `pip install 'any-agent[serve]'` to use this method."
    raise ImportError(msg) from e


__all__ = ["_get_a2a_server", "serve_a2a"]
