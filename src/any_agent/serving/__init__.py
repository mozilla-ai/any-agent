try:
    from .server import A2AServerAsync, _get_a2a_server, _get_a2a_server_async

    __all__ = ["A2AServerAsync", "_get_a2a_server", "_get_a2a_server_async"]

    serving_available = True
except ImportError:
    serving_available = False
