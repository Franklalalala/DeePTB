"""Shared loader for DeePTB's in-repo CUDA extensions.

Use the optional ``so2_cuda_ops`` implementation when installed.  The local
loader remains available so legacy production kernels do not depend on that
extra merely for their build plumbing.
"""

from __future__ import annotations

import importlib
import os
import site
from pathlib import Path
from typing import Iterable, Optional, Sequence


def _load_external_backend():
    try:
        return importlib.import_module("so2_cuda_ops._extension_loader")
    except ModuleNotFoundError as exc:
        # Fall back only when the optional top-level package itself is absent.
        # A missing loader submodule or transitive dependency means an installed
        # backend is broken and must not be hidden behind an unrelated JIT build.
        if exc.name != "so2_cuda_ops":
            raise
        return None


_backend = _load_external_backend()

if _backend is None:
    from torch.utils.cpp_extension import load

    _FALSE = {
        "",
        "0",
        "false",
        "False",
        "FALSE",
        "off",
        "OFF",
        "no",
        "No",
    }

    def truthy_env(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default) not in _FALSE

    def _as_path_strings(
        paths: Optional[Iterable[str | Path]],
    ) -> list[str]:
        if not paths:
            return []
        return [str(Path(path)) for path in paths]

    def _env_cuda_paths() -> tuple[list[str], list[str]]:
        include_paths = []
        library_paths = []
        for site_root in site.getsitepackages():
            root = Path(site_root) / "nvidia"
            for package in ("cuda_runtime", "cublas"):
                include_dir = root / package / "include"
                library_dir = root / package / "lib"
                if include_dir.is_dir():
                    include_paths.append(str(include_dir))
                if library_dir.is_dir():
                    library_paths.append(str(library_dir))
        return include_paths, library_paths

    def load_cuda_extension(
        *,
        name: str,
        source_files: Sequence[str | Path],
        build_dir_env: str,
        default_build_dir: str | Path,
        extra_cflags: Optional[list[str]] = None,
        extra_cuda_cflags: Optional[list[str]] = None,
        extra_ldflags: Optional[list[str]] = None,
        extra_include_paths: Optional[Iterable[str | Path]] = None,
        verbose_env: Optional[str] = None,
    ):
        """Load one CUDA extension using DeePTB's standard conventions."""

        build_dir = Path(
            os.environ.get(build_dir_env, str(default_build_dir))
        )
        build_dir.mkdir(parents=True, exist_ok=True)
        env_include_paths, env_library_paths = _env_cuda_paths()
        include_paths = env_include_paths + _as_path_strings(
            extra_include_paths
        )
        ldflags = [
            f"-L{path}" for path in env_library_paths
        ] + (extra_ldflags or [])
        return load(
            name=name,
            sources=_as_path_strings(source_files),
            extra_cflags=extra_cflags or ["-O3"],
            extra_cuda_cflags=extra_cuda_cflags or ["-O3"],
            extra_ldflags=ldflags,
            extra_include_paths=include_paths,
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=(
                truthy_env(verbose_env, "0") if verbose_env else False
            ),
        )
else:
    globals().update(
        {
            name: getattr(_backend, name)
            for name in dir(_backend)
            if not name.startswith("__")
        }
    )
