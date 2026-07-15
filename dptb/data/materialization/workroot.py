"""Disjoint-root guard shared by the durable materializers.

Both CLIs refuse to run when the protected input root and the work/output root
overlap (either being a parent of, or equal to, the other), and both implement
the same overwrite/resume state machine.  The two tools differ only in the
wording of their error messages, so those are injected by the caller while the
control flow lives here once.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def guard_work_root(
    protected_root: Path | str,
    work_root: Path | str,
    *,
    overwrite: bool,
    resume: bool,
    disjoint_message: str,
    mutually_exclusive_message: str,
    missing_work_root_message: str,
) -> None:
    """Validate/prepare ``work_root`` relative to a protected input root.

    * ``protected_root`` and ``work_root`` must be disjoint (neither contains
      the other) -> ``ValueError(disjoint_message)``.
    * ``overwrite`` and ``resume`` are mutually exclusive ->
      ``ValueError(mutually_exclusive_message)``.
    * ``resume`` requires an existing work root ->
      ``FileNotFoundError(missing_work_root_message)``.
    * otherwise the work root is created, refusing to clobber an existing one
      unless ``overwrite`` is set (in which case it is removed first).
    """

    protected_root = Path(protected_root).resolve()
    work_root = Path(work_root).resolve()
    if (
        protected_root == work_root
        or protected_root in work_root.parents
        or work_root in protected_root.parents
    ):
        raise ValueError(disjoint_message)
    if overwrite and resume:
        raise ValueError(mutually_exclusive_message)
    if resume:
        if not work_root.is_dir():
            raise FileNotFoundError(missing_work_root_message)
        return
    if work_root.exists():
        if not overwrite:
            raise FileExistsError(work_root)
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
