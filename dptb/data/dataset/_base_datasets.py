"""Minimal dataset base used by the maintained LMDB backend."""

import hashlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import yaml

import dptb
from dptb.data import AtomicDataDict
from dptb.utils.torch_geometric import Dataset

from ..transforms import TypeMapper


_DYNAMIC_BATCH_COST_KEYS = ("block", "edge")


def _shape0(value: Any) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        return int(value.shape[0]) if value.ndim >= 1 else 1
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


def _index_count(value: Any) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.ndim >= 2:
            return int(value.shape[-1])
        if value.ndim == 1:
            return int(value.shape[0])
        return 1
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) >= 2:
            return int(shape[-1])
        if len(shape) == 1:
            return int(shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


def _get_value(data: Any, key: str, default=None):
    if isinstance(data, dict):
        return data.get(key, default)
    try:
        return data[key]
    except Exception:
        return getattr(data, key, default)


def _normalize_dynamic_batch_cost_parts(
    parts: Dict[str, Any],
) -> Dict[str, int]:
    out = {key: 0 for key in _DYNAMIC_BATCH_COST_KEYS}
    for key, value in dict(parts or {}).items():
        if key in out:
            out[key] = int(value)
    return out


def _dynamic_batch_parts_from_data(data: Any) -> Dict[str, int]:
    edge = _index_count(
        _get_value(data, AtomicDataDict.EDGE_INDEX_KEY, None)
    )
    if edge <= 0:
        edge = _shape0(
            _get_value(data, AtomicDataDict.EDGE_FEATURES_KEY, None)
        )
    if edge <= 0:
        edge = _shape0(
            _get_value(data, AtomicDataDict.EDGE_H0_KEY, None)
        )

    return _normalize_dynamic_batch_cost_parts(
        {
            "block": 0,
            "edge": max(0, edge),
        }
    )


class AtomicDataset(Dataset):
    """Base contract for the maintained lazy LMDB dataset."""

    root: str
    dtype: torch.dtype

    def __init__(
        self,
        root: str,
        type_mapper: Optional[TypeMapper] = None,
    ):
        self.dtype = torch.get_default_dtype()
        self._dynamic_batch_cost_version = 0
        super().__init__(root=root, transform=type_mapper)

    @property
    def dynamic_batch_cost_version(self) -> int:
        return int(getattr(self, "_dynamic_batch_cost_version", 0))

    def invalidate_dynamic_batch_costs(self) -> None:
        self._dynamic_batch_cost_version = (
            self.dynamic_batch_cost_version + 1
        )

    def _resolve_dynamic_batch_index(self, idx: int) -> int:
        return int(self.indices()[int(idx)])

    def get_dynamic_batch_cost_parts(
        self, idx: int
    ) -> Dict[str, int]:
        return _dynamic_batch_parts_from_data(self[int(idx)])

    def statistics(
        self,
        fields: List[Union[str, Callable]],
        modes: List[str],
        stride: int = 1,
        unbiased: bool = True,
        kwargs: Optional[Dict[str, dict]] = None,
    ) -> List[tuple]:
        del fields, modes, stride, unbiased, kwargs
        raise NotImplementedError(
            "The lazy LMDB backend does not implement generic statistics."
        )

    @property
    def type_mapper(self) -> Optional[TypeMapper]:
        return self.transform

    def _get_parameters(self) -> Dict[str, Any]:
        """Return constructor parameters that identify the dataset cache."""

        parameter_names = list(inspect.signature(self.__init__).parameters)
        ignored = {"type_mapper", "root"}
        params = {
            key: getattr(self, key)
            for key in parameter_names
            if key not in ignored and hasattr(self, key)
        }
        params["dtype"] = str(self.dtype)
        params["dptb_version"] = dptb.__version__
        return params

    @property
    def processed_dir(self) -> str:
        buffer = yaml.dump(self._get_parameters()).encode("ascii")
        parameter_hash = hashlib.sha1(buffer).hexdigest()
        return f"{self.root}/processed_dataset_{parameter_hash}"
