"""Strict-reload output auditor for the durable materializers.

After a cache is written, the dual-prior materializer re-opens every output
split through the real ``LMDBDataset`` loader once per prior kind, forcing the
loader's own fail-closed contract to re-validate every committed record.  The
loader construction / strict configuration / teardown / heartbeat callables are
injected so this stays decoupled from any specific dataset implementation and
so a CLI's module-level (monkeypatchable) helpers remain the integration point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Tuple


class DatasetAuditor:
    """Re-materialize output splits through the strict dataset loader.

    Parameters are callables so the auditor is agnostic to the concrete dataset
    type:

    * ``build_context_dataset(input_json: str, split_root: str) -> (dataset, _)``
    * ``configure_strict_dataset(dataset, *, kind: str, source: str) -> None``
    * ``close_dataset(dataset) -> None``
    * ``heartbeat(work_root, **payload) -> None``
    """

    def __init__(
        self,
        *,
        build_context_dataset: Callable[[str, str], Tuple[Any, Any]],
        configure_strict_dataset: Callable[..., None],
        close_dataset: Callable[[Any], None],
        heartbeat: Callable[..., None],
    ):
        self._build = build_context_dataset
        self._configure = configure_strict_dataset
        self._close = close_dataset
        self._heartbeat = heartbeat

    def audit(
        self,
        *,
        input_json: Path | str,
        split_roots: Mapping[str, Path | str],
        prior_sources: Sequence[Tuple[str, str]],
        expected_entries: Mapping[str, int],
        work_root: Path | str,
        stage: str = "strict_audit",
    ) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for split, split_root in split_roots.items():
            split_result: dict[str, int] = {}
            for kind, source in prior_sources:
                dataset, _ = self._build(str(input_json), str(split_root))
                try:
                    if len(dataset) != int(expected_entries[split]):
                        raise ValueError(f"Output audit {split}: dataset length mismatch.")
                    self._configure(dataset, kind=kind, source=source)
                    for index in range(len(dataset)):
                        dataset.get(index)
                        self._heartbeat(
                            work_root,
                            stage=stage,
                            split=split,
                            prior_kind=kind,
                            completed=index + 1,
                            total=len(dataset),
                        )
                    split_result[kind] = len(dataset)
                finally:
                    self._close(dataset)
            result[split] = split_result
        return result
