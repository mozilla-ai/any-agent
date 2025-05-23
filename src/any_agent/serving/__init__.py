import sys

if sys.version_info < (3, 13):
    msg = "Serving with A2A requires Python 3.13 or higher! 🐍✨"
    raise RuntimeError(msg)

try:
    from .server import _get_a2a_app, serve_a2a, serve_a2a_async
except ImportError as e:
    msg = "You need to `pip install 'any-agent[serve]'` to use this method."
    raise ImportError(msg) from e


__all__ = ["_get_a2a_app", "serve_a2a", "serve_a2a_async"]
