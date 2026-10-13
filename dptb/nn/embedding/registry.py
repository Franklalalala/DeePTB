"""Registry that maps ``model_options.embedding.method`` names to constructors."""
from dptb.utils.register import Register


class Embedding:
    """Construct the embedding registered under ``method``.

    Implementations register themselves with ``@Embedding.register("name")``
    when :mod:`dptb.nn.embedding` imports them.
    """

    _register = Register()

    @staticmethod
    def register(target):
        return Embedding._register.register(target)

    def __new__(cls, method: str, **options):
        if method in {"lem", "slem"}:
            from dptb.nn.conditioning.inputs import resolve_legacy_prior_method

            method = resolve_legacy_prior_method(method, options)
        if method not in Embedding._register:
            raise ValueError(f"Embedding method {method!r} is not registered.")
        return Embedding._register[method](**options)
