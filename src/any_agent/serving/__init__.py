try:
    from .server import A2AServer, _get_a2a_server

    __all__ = ["A2AServer", "_get_a2a_server"]

    a2a_serving_available = True
except ImportError:
    __all__ = []

    a2a_serving_available = False
