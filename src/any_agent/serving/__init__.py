try:
    from .config import A2AServingConfig
    from .server import _get_a2a_app, _get_a2a_app_async, serve_a2a, serve_a2a_async
except ImportError as e:
    msg = "You need to `pip install 'any-agent[a2a]'` to use this method."
    raise ImportError(msg) from e


__all__ = [
    "A2AServingConfig",
    "_get_a2a_app",
    "_get_a2a_app_async",
    "serve_a2a",
    "serve_a2a_async",
]
