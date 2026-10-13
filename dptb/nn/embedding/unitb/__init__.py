"""UniTB embeddings and their shared message-passing implementation."""

__all__ = ["UniTB"]


def __getattr__(name):
    if name == "UniTB":
        from .model import UniTB
        globals()[name] = UniTB
        return UniTB
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
