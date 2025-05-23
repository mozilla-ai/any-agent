import sys

if sys.version_info < (3, 13):
    raise RuntimeError("Serving with A2A requires Python 3.13 or higher! 🐍✨")

try:
    from .server import _get_a2a_server, serve_a2a
except ImportError as e:
    msg = "You need to `pip install 'any-agent[serve]'` to use this method."
    raise ImportError(msg) from e


__all__ = ["_get_a2a_server", "serve_a2a"]
