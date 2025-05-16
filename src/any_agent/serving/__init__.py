try:
    from .server import _get_a2a_server, _get_a2a_server_async, A2AServerAsync

    __all__ = ["_get_a2a_server", "_get_a2a_server_async", "A2AServerAsync"]

    serving_available = True
except ImportError:
    serving_available = False
