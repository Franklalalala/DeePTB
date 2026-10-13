"""Separate immutable package resources from explicitly selected work stores."""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def work_root():
    """Require a writable task directory instead of writing into an installation."""
    value = os.environ.get("H0_WORK_ROOT")
    if not value:
        raise ValueError("Set H0_WORK_ROOT to an external writable task directory")
    return Path(value).expanduser().resolve()


def table_root():
    """Use an explicit prepared store, with a work-directory fallback."""
    value = os.environ.get("H0_OFFLINE_TABLE_DIR")
    return Path(value).expanduser().resolve() if value else work_root() / "offline_tables"
