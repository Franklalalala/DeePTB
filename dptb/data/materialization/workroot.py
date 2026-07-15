"""Disjoint-root guard and advisory work-root lock for the durable materializers.

Both CLIs refuse to run when the protected input root and the work/output root
overlap (either being a parent of, or equal to, the other), and both implement
the same overwrite/resume state machine.  The two tools differ only in the
wording of their error messages, so those are injected by the caller while the
control flow lives here once.

On top of the static guard, :class:`WorkRootLock` provides an *advisory* lock
file inside the work root so two concurrent materializer processes cannot
interleave writes into the same output tree:

* acquiring writes ``workroot.lock.json`` (run UUID + pid + host + timestamp)
  with ``O_CREAT | O_EXCL`` + fsync;
* a live foreign lock refuses the run (:class:`WorkRootLockError`);
* a stale lock (dead pid on the same host, or older than the configurable
  stale age) is taken over;
* the lock is released on clean exit (context-manager exit), and only if it
  still carries this run's UUID.  A hard crash leaves the file behind, which
  the stale-age policy then reclaims.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

# Lock file name inside the work root, and the default age after which a lock
# whose liveness cannot be proven (foreign host / unreadable payload) is
# considered abandoned.
LOCK_FILE_NAME = "workroot.lock.json"
LOCK_SCHEMA = "deeptb.materialization_workroot_lock/v1"
DEFAULT_LOCK_STALE_SECONDS = 24.0 * 3600.0


class WorkRootLockError(RuntimeError):
    """Raised when a live advisory lock already owns the work root."""


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness probe for a pid on *this* host.

    ``os.kill(pid, 0)`` performs an existence check on POSIX and Windows
    alike.  ``PermissionError`` proves the pid exists; look-up failures (and
    the Windows ``OSError`` flavors for a vanished pid) mean it is gone.
    """

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_lock_payload(lock_path: Path) -> Optional[dict]:
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lock_age_seconds(lock_path: Path, payload: Optional[Mapping[str, Any]]) -> float:
    """Age of the lock, preferring the embedded timestamp over file mtime."""

    acquired = None
    if payload is not None:
        try:
            acquired = float(payload["acquired_unix"])
        except (KeyError, TypeError, ValueError):
            acquired = None
    if acquired is None:
        try:
            acquired = lock_path.stat().st_mtime
        except OSError:
            # The lock vanished while we were looking at it: not live.
            return float("inf")
    return max(0.0, time.time() - acquired)


def _lock_is_live(
    lock_path: Path,
    payload: Optional[Mapping[str, Any]],
    *,
    stale_after_seconds: float,
) -> bool:
    """Decide whether an existing lock file still protects the work root.

    * same host + owner pid alive  -> live (age does not matter);
    * same host + owner pid dead   -> stale (immediate takeover allowed);
    * foreign host / unreadable    -> live until older than the stale age.
    """

    if payload is not None:
        host = str(payload.get("host", ""))
        pid_raw = payload.get("pid")
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            pid = -1
        if host and host == socket.gethostname() and pid > 0:
            return _pid_alive(pid)
    return _lock_age_seconds(lock_path, payload) < float(stale_after_seconds)


def describe_lock(lock_path: Path, payload: Optional[Mapping[str, Any]]) -> str:
    """Human-readable one-line description of an existing lock file."""

    if payload is None:
        return f"{lock_path} (unreadable payload)"
    return (
        f"{lock_path} (run_uuid={payload.get('run_uuid')!r}, "
        f"pid={payload.get('pid')!r}, host={payload.get('host')!r}, "
        f"acquired_unix={payload.get('acquired_unix')!r})"
    )


def check_work_root_lock(
    work_root: Path | str,
    *,
    stale_after_seconds: float | None = None,
) -> None:
    """Raise :class:`WorkRootLockError` if a LIVE advisory lock exists.

    A missing lock file, a lock owned by a dead same-host pid, or a lock older
    than the stale age all pass silently (the stale file is left for
    :meth:`WorkRootLock.acquire` to take over or for ``rmtree`` to remove).
    """

    stale = (
        DEFAULT_LOCK_STALE_SECONDS
        if stale_after_seconds is None
        else float(stale_after_seconds)
    )
    lock_path = Path(work_root) / LOCK_FILE_NAME
    if not lock_path.exists():
        return
    payload = _read_lock_payload(lock_path)
    if _lock_is_live(lock_path, payload, stale_after_seconds=stale):
        raise WorkRootLockError(
            f"work root {Path(work_root)} is locked by a live materializer run: "
            f"{describe_lock(lock_path, payload)}. Refusing to run concurrently; "
            f"if that process is truly gone, remove the lock file or wait for "
            f"the stale age ({stale:.0f}s)."
        )


class WorkRootLock:
    """Advisory per-work-root lock (context manager).

    ``acquire()`` atomically creates the lock file (``O_CREAT | O_EXCL``) with
    this run's UUID/pid/host/timestamp and fsyncs it.  If a lock already
    exists it is either refused (live) or taken over (stale) exactly once.
    ``release()`` removes the file only while it still carries this run's
    UUID, so a takeover by a later run is never clobbered.
    """

    def __init__(
        self,
        work_root: Path | str,
        *,
        stale_after_seconds: float | None = None,
        run_uuid: str | None = None,
    ) -> None:
        self.work_root = Path(work_root)
        self.lock_path = self.work_root / LOCK_FILE_NAME
        self.stale_after_seconds = (
            DEFAULT_LOCK_STALE_SECONDS
            if stale_after_seconds is None
            else float(stale_after_seconds)
        )
        self.run_uuid = str(run_uuid) if run_uuid else uuid.uuid4().hex
        self._acquired = False

    def _payload(self) -> dict:
        return {
            "schema": LOCK_SCHEMA,
            "run_uuid": self.run_uuid,
            "pid": int(os.getpid()),
            "host": socket.gethostname(),
            "acquired_unix": time.time(),
            "stale_after_seconds": self.stale_after_seconds,
        }

    def acquire(self) -> "WorkRootLock":
        if self._acquired:
            return self
        if not self.work_root.is_dir():
            raise FileNotFoundError(
                f"cannot lock non-existent work root {self.work_root}"
            )
        # Two attempts: create; on FileExistsError evaluate liveness and (only
        # for a stale lock) remove it once and retry the exclusive create.
        for attempt in range(2):
            try:
                fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                payload = _read_lock_payload(self.lock_path)
                if _lock_is_live(
                    self.lock_path,
                    payload,
                    stale_after_seconds=self.stale_after_seconds,
                ):
                    raise WorkRootLockError(
                        f"work root {self.work_root} is locked by a live "
                        f"materializer run: {describe_lock(self.lock_path, payload)}. "
                        "Refusing to run concurrently."
                    )
                if attempt == 1:
                    break
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(self._payload(), indent=2, sort_keys=True) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                try:
                    self.lock_path.unlink()
                except OSError:
                    pass
                raise
            self._acquired = True
            return self
        raise WorkRootLockError(
            f"could not acquire {self.lock_path} even after removing a stale "
            "lock; another run is racing for this work root."
        )

    def release(self) -> None:
        """Remove the lock file iff it still belongs to this run (best effort)."""

        if not self._acquired:
            return
        self._acquired = False
        try:
            payload = _read_lock_payload(self.lock_path)
            if payload is not None and payload.get("run_uuid") == self.run_uuid:
                self.lock_path.unlink()
        except OSError:
            pass

    @property
    def acquired(self) -> bool:
        return self._acquired

    def __enter__(self) -> "WorkRootLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def guard_work_root(
    protected_root: Path | str,
    work_root: Path | str,
    *,
    overwrite: bool,
    resume: bool,
    disjoint_message: str,
    mutually_exclusive_message: str,
    missing_work_root_message: str,
    lock_stale_after_seconds: float | None = None,
) -> None:
    """Validate/prepare ``work_root`` relative to a protected input root.

    * ``protected_root`` and ``work_root`` must be disjoint (neither contains
      the other) -> ``ValueError(disjoint_message)``.
    * ``overwrite`` and ``resume`` are mutually exclusive ->
      ``ValueError(mutually_exclusive_message)``.
    * ``resume`` requires an existing work root ->
      ``FileNotFoundError(missing_work_root_message)``.
    * an existing work root owned by a LIVE advisory lock refuses both resume
      and overwrite -> :class:`WorkRootLockError`.  The check runs BEFORE any
      destructive ``rmtree`` so a concurrent run's tree is never deleted.
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
        check_work_root_lock(
            work_root, stale_after_seconds=lock_stale_after_seconds
        )
        return
    if work_root.exists():
        if not overwrite:
            raise FileExistsError(work_root)
        check_work_root_lock(
            work_root, stale_after_seconds=lock_stale_after_seconds
        )
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
