try:
    from .server import _get_a2a_app, serve_a2a, serve_a2a_async
except ImportError as e:
    msg = "You need to `pip install 'any-agent[a2a]'` to use this method."
    raise ImportError(msg) from e


__all__ = ["_get_a2a_app", "serve_a2a", "serve_a2a_async"]
