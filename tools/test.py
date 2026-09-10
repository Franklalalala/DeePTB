"""Run a short behavioral smoke suite, or explicit pytest targets supplied by the caller."""

from pathlib import Path
import os
import sys

SMOKE = [
    "dptb/tests/test_loopscf_numerics.py",
    "dptb/tests/test_band_losses.py",
    "dptb/tests/test_record_codec.py",
    "dptb/tests/test_spectral_records.py",
    "dptb/tests/test_prior2b_pa.py",
    "dptb/tests/test_distance_ensemble_stitch.py",
]


def main():
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import torch
    import pytest

    torch.set_num_threads(1)
    args = sys.argv[1:]
    has_target = any(
        not arg.startswith("-")
        and (Path(arg.split("::", 1)[0]).is_dir() or ".py" in arg.split("::", 1)[0])
        for arg in args
    )
    targets = args if has_target else [*SMOKE, *args]
    return pytest.main(["-q", "--tb=short", *targets])


if __name__ == "__main__":
    raise SystemExit(main())
