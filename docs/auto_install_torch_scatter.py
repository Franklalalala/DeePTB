from __future__ import annotations

import re
import subprocess
import sys
from typing import Optional


def _torch_wheel_version(torch_version: str) -> str:
    """Map a PyTorch patch release to the PyG wheel index version.

    PyG publishes one wheel page per PyTorch minor line (for example PyTorch
    2.5.x uses ``torch-2.5.0+...html``), not one page per patch release.
    """

    public = str(torch_version).split("+", 1)[0]
    match = re.match(r"^(\d+)\.(\d+)", public)
    if match is None:
        raise ValueError(f"Cannot derive a PyG wheel index from torch {torch_version!r}.")
    return f"{int(match.group(1))}.{int(match.group(2))}.0"


def _platform_tag(
    torch_version: str,
    cuda_version: Optional[str],
    hip_version: Optional[str],
) -> str:
    local = str(torch_version).partition("+")[2]
    if local.startswith("rocm") or hip_version:
        raise RuntimeError(
            "ROCm torch-scatter wheels are not hosted on the standard PyG "
            "wheel index; install a matching ROCm build explicitly."
        )
    if local == "cpu" or local.startswith("cu"):
        return local
    if cuda_version:
        return "cu" + str(cuda_version).replace(".", "")
    return "cpu"


def torch_scatter_wheel_url(
    torch_version: str,
    cuda_version: Optional[str] = None,
    hip_version: Optional[str] = None,
) -> str:
    version = _torch_wheel_version(torch_version)
    platform = _platform_tag(torch_version, cuda_version, hip_version)
    return f"https://data.pyg.org/whl/torch-{version}+{platform}.html"


def main() -> int:
    try:
        import torch
    except ImportError:
        print("The torch module is not found; install PyTorch first.", file=sys.stderr)
        return 1

    print(f"Current torch version: {torch.__version__}")
    print(f"CUDA used by PyTorch: {torch.version.cuda or 'cpu'}")

    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        url = torch_scatter_wheel_url(
            torch.__version__,
            cuda_version=torch.version.cuda,
            hip_version=getattr(torch.version, "hip", None),
        )
        print(f"torch-scatter will be installed from {url}...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch-scatter==2.1.2",
                "-f",
                url,
            ],
            check=True,
        )
        print("Installation complete.")
    else:
        print("torch-scatter is already installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
