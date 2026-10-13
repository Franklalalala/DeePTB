"""Public construction API for every DeePTB model family.

Use ``build_model`` for a complete Hamiltonian model and ``build_embedding``
for a descriptor. Imports are resolved on demand so inspecting the catalogue
does not initialize unrelated model families or optional backends.
"""

from importlib import import_module

_EXPORTS = {
    "build_model": (".build", "build_model"),
    "build_embedding": (".embedding", "build_embedding"),
    "list_embeddings": (".embedding", "list_embeddings"),
    "NNENV": (".deeptb", "NNENV"),
    "MIX": (".deeptb", "MIX"),
    "NNSK": (".nnsk", "NNSK"),
    "DFTBSK": (".dftbsk", "DFTBSK"),
    "E3Hamiltonian": (".hamiltonian", "E3Hamiltonian"),
    "SKHamiltonian": (".hamiltonian", "SKHamiltonian"),
    "HR2HK": (".hr2hk", "HR2HK"),
    "HR2HK_Gamma_Only": (".hr2hk", "HR2HK_Gamma_Only"),
    "Eigenvalues": (".energy", "Eigenvalues"),
    "Eigh": (".energy", "Eigh"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
