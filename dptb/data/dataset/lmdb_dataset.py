import numpy as np
from typing import Tuple, Dict, Any, List, Callable, Union, Optional

import torch
from dptb.utils.tools import download_url, extract_zip

import os
import os.path as osp
import glob
from dptb.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
)
from tqdm import tqdm
from ..transforms import TypeMapper
from ._base_datasets import (
    AtomicDataset,
    _dynamic_batch_parts_from_data,
)
from dptb.nn.hamiltonian import E3Hamiltonian
import lmdb
from dptb.data.interfaces.ham_to_feature import block_to_feature
import pickle


def _parse_lmdb_block_key(key: Any):
    if not isinstance(key, str):
        return None
    parts = key.split("_")
    if len(parts) != 5:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def _count_offsite_lmdb_blocks(blocks: Any) -> int:
    if not isinstance(blocks, dict):
        return 0

    count = 0
    for key in blocks.keys():
        parsed = _parse_lmdb_block_key(key)
        if parsed is None:
            continue
        i, j, rx, ry, rz = parsed
        if i == j and rx == 0 and ry == 0 and rz == 0:
            continue
        count += 1
    return count


def _read_lmdb_entry(path: str, index: int):
    db_env = lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=2048,
    )
    try:
        with db_env.begin(buffers=True) as txn:
            data = txn.get(int(index).to_bytes(length=4, byteorder='big'))
            if data is None:
                raise IndexError(f"LMDB entry {index} not found in {path}")
            return pickle.loads(bytes(data))
    finally:
        db_env.close()


def _lmdb_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _lmdb_scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    array = np.asarray(value)
    if array.shape == ():
        return bool(array.item())
    return bool(value)


def _lmdb_scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    array = np.asarray(value)
    if array.shape == ():
        return int(array.item())
    return int(value)


def _soc_uureal_keep_mask(
    data_dict: Dict[str, Any],
    full_rme: int,
    keep_mask: Optional[Any] = None,
) -> torch.Tensor:
    if keep_mask is not None:
        mask = torch.as_tensor(keep_mask, dtype=torch.bool).flatten()
    else:
        keep = data_dict.get("soc_uureal_keep", None)
        if keep is None:
            raise ValueError(
                "Compact SOC uu_real LMDB entry is missing soc_uureal_keep metadata."
            )
        keep_tensor = torch.as_tensor(keep)
        if keep_tensor.ndim == 0:
            keep_count = int(keep_tensor.item())
            if keep_count == full_rme:
                mask = torch.ones(full_rme, dtype=torch.bool)
            else:
                raise ValueError(
                    "Compact SOC uu_real LMDB entry stores only a keep count. "
                    "Enable nextham_uureal_mask so the dataset can use the "
                    "type_mapper.mask_uureal layout mask for expansion."
                )
        elif keep_tensor.dtype == torch.bool:
            mask = keep_tensor.flatten()
        else:
            keep_flat = keep_tensor.flatten().to(dtype=torch.long)
            if keep_flat.numel() == full_rme and (
                keep_flat.numel() == 0 or int(keep_flat.max().item()) <= 1
            ):
                mask = keep_flat.to(dtype=torch.bool)
            else:
                mask = torch.zeros(full_rme, dtype=torch.bool)
                mask[keep_flat] = True

    if mask.numel() != full_rme:
        raise ValueError(
            "Compact SOC uu_real mask width does not match full RME: "
            f"mask={mask.numel()}, full_rme={full_rme}."
        )
    return mask


def _expand_soc_uureal_compact(
    value: Any,
    data_dict: Dict[str, Any],
    field_name: str,
    keep_mask: Optional[Any] = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not _lmdb_scalar_bool(data_dict.get("soc_uureal_compact", False)):
        return tensor

    full_rme_value = data_dict.get("soc_uureal_full_rme", None)
    if full_rme_value is None:
        raise ValueError(
            f"Compact SOC uu_real LMDB field {field_name} is missing "
            "soc_uureal_full_rme metadata."
        )
    full_rme = _lmdb_scalar_int(full_rme_value)
    if full_rme <= 0:
        raise ValueError(
            f"Compact SOC uu_real LMDB field {field_name} has invalid "
            f"soc_uureal_full_rme={full_rme}."
        )

    if keep_mask is not None:
        target_mask = torch.as_tensor(keep_mask, dtype=torch.bool).flatten()
        target_rme = int(target_mask.numel())
        if target_rme != full_rme and bool(target_mask.all().item()):
            if tensor.shape[-1] == target_rme:
                return tensor
            raise ValueError(
                f"Compact SOC uu_real LMDB field {field_name} has width "
                f"{tensor.shape[-1]}; reduced uu_real target expects compact "
                f"width {target_rme}."
            )

    if tensor.shape[-1] == full_rme:
        return tensor

    mask = _soc_uureal_keep_mask(data_dict, full_rme, keep_mask=keep_mask)
    compact_rme = int(mask.sum().item())
    if tensor.shape[-1] != compact_rme:
        raise ValueError(
            f"Compact SOC uu_real LMDB field {field_name} has incompatible "
            f"width {tensor.shape[-1]}; expected compact width {compact_rme} "
            f"or full width {full_rme}."
        )

    expanded = torch.zeros(
        (*tensor.shape[:-1], full_rme),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    expanded[..., mask.to(device=tensor.device)] = tensor
    return expanded


_ATOMICDATA_CONSTRUCTOR_OPTIONS = {"r_max", "er_max", "oer_max", "self_interaction"}


class LMDBDataset(AtomicDataset):
    prefer_loaded_dynamic_batch_cost_parts = True

    def __init__(
            self,
            root: str,
            info_files: dict,
            url: Optional[str] = None,
            include_frames: Optional[List[int]] = None,
            type_mapper: TypeMapper = None,
            orthogonal: bool = False,
            get_Hamiltonian: bool = False,
            get_H0: bool = False,
            get_overlap: bool = False,
            get_DM: bool = False,
            get_eigenvalues: bool = False,
            h0_key: str = "hamiltonian_0",
            prefer_precomputed_h0: bool = True,
    ):
        # TO DO, this may be simplified
        # See if a subclass defines some inputs
        self.url = getattr(type(self), "URL", url)
        self.include_frames = include_frames
        self.info_files = info_files  # there should be one info file for one LMDB Dataset
        # print(self.info_files)

        self.data = None
        # !!! don't delete this block.
        # otherwise the inherent children class
        # will ignore the download function here
        class_type = type(self)
        if class_type != LMDBDataset:
            if "download" not in self.__class__.__dict__:
                class_type.download = LMDBDataset.download

        # Initialize the InMemoryDataset, which runs download and process
        # See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets
        # Then pre-process the data if disk files are not found
        super().__init__(root=root, type_mapper=type_mapper)  # the type_mapper will be called in getitem in PyG data class
        self.get_Hamiltonian = get_Hamiltonian
        self.get_H0 = get_H0
        self.get_overlap = get_overlap
        self.get_DM = get_DM
        self.get_eigenvalues = get_eigenvalues
        self.orthogonal = orthogonal
        self.h0_key = h0_key
        self.prefer_precomputed_h0 = prefer_precomputed_h0
        assert not get_Hamiltonian * get_DM, "Hamiltonian and Density Matrix can only loaded one at a time, for which will occupy the same attribute in the AtomicData."

        self.num_graphs = 0
        self.file_map = []
        self.index_map = []
        self._lmdb_path_map = []
        self._lmdb_env_cache = {}
        self._dynamic_batch_cost_parts_cache = {}
        for file in self.info_files.keys():
            lmdb_paths = self.simple_get_lmdb_path(file)
            for lmdb_path in lmdb_paths:
                db_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048)
                with db_env.begin(buffers=True) as txn:
                    self.num_graphs += txn.stat()['entries']
                    self.file_map += [file] * txn.stat()['entries']
                    self.index_map += list(range(txn.stat()['entries']))
                    self._lmdb_path_map += [lmdb_path] * txn.stat()['entries']
                db_env.close()

    def len(self):
        return self.num_graphs

    def simple_get_lmdb_path(self, folder_name: str):
        """
        Finds LMDB directory paths matching the given folder name under root path(s).
        Supports wildcards in root paths and returns all existing matches.

        Args:
            folder_name: Folder name (or path). Only the base name is used for matching.

        Returns:
            list[str]: List of existing LMDB paths. Empty list if none found.

        Notes:
            - Uses only the base name of `folder_name` (e.g., "data" from "/path/to/data")
            - Processes wildcards (*, ?, []) in root paths via `glob`
            - Handles both single root (str) and multiple roots (list)
        """
        folder_name = os.path.split(folder_name)[-1]  # Keep only base name

        # Normalize root paths to list for consistent processing
        root_paths = [self.root] if isinstance(self.root, str) else self.root
        candidate_paths = []

        for root_path in root_paths:
            abs_path = os.path.abspath(root_path)

            # Handle wildcard-containing roots
            if any(char in abs_path for char in ['*', '?', '[']):
                for expanded_path in glob.glob(abs_path):
                    if os.path.isdir(expanded_path):
                        candidate_paths.append(os.path.join(expanded_path, folder_name))
            # Standard path processing
            else:
                candidate_paths.append(os.path.join(abs_path, folder_name))

        # Return all existing paths
        return [path for path in candidate_paths if os.path.exists(path)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lmdb_env_cache"] = {}
        return state

    def invalidate_dynamic_batch_costs(self) -> None:
        super().invalidate_dynamic_batch_costs()
        self._dynamic_batch_cost_parts_cache = {}

    def __del__(self):
        for env in getattr(self, "_lmdb_env_cache", {}).values():
            try:
                env.close()
            except Exception:
                pass

    def _get_lmdb_env(self, path: str):
        cache = getattr(self, "_lmdb_env_cache", None)
        if cache is None:
            cache = {}
            self._lmdb_env_cache = cache
        env = cache.get(path)
        if env is None:
            env = lmdb.open(
                path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048,
            )
            cache[path] = env
        return env

    def _load_data_dict(self, idx: int):
        lmdb_paths = getattr(self, "_lmdb_path_map", None)
        if lmdb_paths is not None and len(lmdb_paths) == len(self.index_map):
            candidate_paths = [lmdb_paths[idx]]
        else:
            candidate_paths = self.simple_get_lmdb_path(self.file_map[idx])

        key = self.index_map[int(idx)].to_bytes(length=4, byteorder='big')
        for lmdb_path in candidate_paths:
            env = self._get_lmdb_env(lmdb_path)
            with env.begin(buffers=True) as txn:
                data = txn.get(key)
                if data is not None:
                    return pickle.loads(bytes(data))
        raise IndexError(f"LMDB entry {self.index_map[int(idx)]} not found for dataset index {idx}")

    def get_dynamic_batch_cost_parts(self, idx: int) -> Dict[str, int]:
        raw_idx = self._resolve_dynamic_batch_index(idx)
        cache = getattr(self, "_dynamic_batch_cost_parts_cache", None)
        if cache is None:
            cache = {}
            self._dynamic_batch_cost_parts_cache = cache
        if raw_idx in cache:
            return dict(cache[raw_idx])

        data_dict = self._load_data_dict(raw_idx)
        parts = _dynamic_batch_parts_from_data(data_dict)
        block_keys = [
            "hamiltonian",
            getattr(self, "h0_key", "hamiltonian_0"),
            "hamiltonian_0",
            "density_matrix",
            "overlap",
        ]
        for key in dict.fromkeys(block_keys):
            block_count = _count_offsite_lmdb_blocks(data_dict.get(key, None))
            if block_count > 0:
                parts["block"] = block_count
                break
        cache[raw_idx] = dict(parts)
        return parts

    @property
    def raw_file_names(self):
        # TODO: this is not implemented.
        # need to give a valid path so the download would not be triggered
        return ["data.mdb", "lock.mdb"]

    @property
    def raw_dir(self):
        return self.root

    def download(self):
        if (not hasattr(self, "url")) or (self.url is None):
            # Don't download, assume present. Later could have FileNotFound if the files don't actually exist
            pass
        else:
            download_path = download_url(self.url, self.raw_dir)
            if download_path.endswith(".zip"):
                extract_zip(download_path, self.raw_dir)

    def get(self, idx):
        data_dict = self._load_data_dict(idx)
        cell, pos, atomic_numbers = \
            data_dict[AtomicDataDict.CELL_KEY], \
                data_dict[AtomicDataDict.POSITIONS_KEY], \
                data_dict[AtomicDataDict.ATOMIC_NUMBERS_KEY]

        pbc = data_dict[AtomicDataDict.PBC_KEY]

        if self.get_Hamiltonian:
            blocks = data_dict.get("hamiltonian", None)
            # kk, vv = blocks.keys(), blocks.values()
            # vv = map(lambda x: np.frombuffer(x, np.float32).reshape, vv)
            # blocks = dict(zip(kk, vv))
            # del kk
            # del vv

        if self.get_overlap:
            overlap = data_dict.get("overlap", None)
            # kk, vv = overlap.keys(), overlap.values()
            # vv = map(lambda x: np.frombuffer(x, np.float32), vv)
            # overlap = dict(zip(kk, vv))
            # del kk
            # del vv
        else:
            overlap = False

        if self.get_DM:
            blocks = data_dict.get("density_matrix", None)
            # kk, vv = blocks.keys(), blocks.values()
            # vv = map(lambda x: np.frombuffer(x, np.float32), vv)
            # blocks = dict(zip(kk, vv))
            # del kk
            # del vv

        if not (self.get_Hamiltonian or self.get_DM):
            blocks = False

        pre_node_features = data_dict.get(AtomicDataDict.NODE_FEATURES_KEY, None)
        pre_edge_features = data_dict.get(AtomicDataDict.EDGE_FEATURES_KEY, None)
        pre_node_overlap = data_dict.get(AtomicDataDict.NODE_OVERLAP_KEY, None)
        pre_edge_overlap = data_dict.get(AtomicDataDict.EDGE_OVERLAP_KEY, None)

        h0_blocks = data_dict.get(self.h0_key, None) if self.get_H0 else None
        node_h0 = data_dict.get(AtomicDataDict.NODE_H0_KEY, None) if self.get_H0 else None
        edge_h0 = data_dict.get(AtomicDataDict.EDGE_H0_KEY, None) if self.get_H0 else None
        soc_uureal_keep_mask = getattr(self.type_mapper, "mask_uureal", None)

        if self.info_files[self.file_map[idx]]['train_dip'] == True:
            self.info_files[self.file_map[idx]].update({'dip': data_dict['dipole_moment']})

        if self.info_files[self.file_map[idx]]['train_w_charge'] == True:
            self.info_files[self.file_map[idx]].update({'charge': np.array(data_dict['charge'])})

        if self.info_files[self.file_map[idx]]['train_w_eps'] == True:
            self.info_files[self.file_map[idx]].update({'dielectric_constant': np.array(data_dict['dielectric_constant'])})
        if self.info_files[self.file_map[idx]]['train_w_homo_lumo_gap'] == True:
            self.info_files[self.file_map[idx]].update({
                'GAP_eV': np.array(data_dict['GAP_eV']),
                'LUMO_eV': np.array(data_dict['LUMO_eV']),
                'HOMO_eV': np.array(data_dict['HOMO_eV']),
            })

        if self.info_files[self.file_map[idx]]['train_polar'] == True:
            self.info_files[self.file_map[idx]].update({'polar': data_dict['polarizability']})

        if self.info_files[self.file_map[idx]]['wave_align'] == True:
            orbital_energies = data_dict.get('orbital_energies', 0)
            orbital_coefficients = data_dict.get('orbital_coefficients', 0)
            self.info_files[self.file_map[idx]].update(
                {'orbital_energies': orbital_energies, 'orbital_coefficients': orbital_coefficients})

        cache_info = {}
        for key in ['train_polar', 'train_dip', 'wave_align', 'train_w_charge', 'train_w_eps', 'train_w_homo_lumo_gap']:
            cache_info.update({key: self.info_files[self.file_map[idx]][key]})
            del self.info_files[self.file_map[idx]][key]

        # transform blocks to atomicdata features, or use precomputed features directly
        need_main_features = bool(self.get_Hamiltonian or self.get_DM)
        need_overlap_features = bool(self.get_overlap)
        has_pre_main = pre_node_features is not None and pre_edge_features is not None
        has_pre_overlap = (not need_overlap_features) or (
            pre_edge_overlap is not None and (self.orthogonal or pre_node_overlap is not None)
        )

        uses_pre_main = bool(has_pre_main and has_pre_overlap)
        uses_pre_h0 = bool(
            self.get_H0
            and self.prefer_precomputed_h0
            and node_h0 is not None
            and edge_h0 is not None
        )
        stored_edge_index = data_dict.get(AtomicDataDict.EDGE_INDEX_KEY, None)
        stored_edge_shift = data_dict.get(AtomicDataDict.EDGE_CELL_SHIFT_KEY, None)
        has_stored_edge_graph = stored_edge_index is not None and stored_edge_shift is not None
        info = self.info_files[self.file_map[idx]]
        needs_missing_env_graph = (
            info.get("er_max", None) is not None
            and data_dict.get(AtomicDataDict.ENV_INDEX_KEY, None) is None
        )
        needs_missing_onsitenv_graph = (
            info.get("oer_max", None) is not None
            and data_dict.get(AtomicDataDict.ONSITENV_INDEX_KEY, None) is None
        )
        use_stored_edge_graph = bool(
            has_stored_edge_graph
            and (uses_pre_main or uses_pre_h0)
            and not needs_missing_env_graph
            and not needs_missing_onsitenv_graph
        )

        if use_stored_edge_graph:
            atomicdata_kwargs = {
                key: value
                for key, value in info.items()
                if key not in _ATOMICDATA_CONSTRUCTOR_OPTIONS
            }
            atomicdata_kwargs[AtomicDataDict.EDGE_INDEX_KEY] = _lmdb_tensor(stored_edge_index, torch.long)
            atomicdata_kwargs[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = _lmdb_tensor(
                stored_edge_shift, torch.get_default_dtype()
            )
            if data_dict.get(AtomicDataDict.ENV_INDEX_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ENV_INDEX_KEY] = _lmdb_tensor(
                    data_dict[AtomicDataDict.ENV_INDEX_KEY], torch.long
                )
            if data_dict.get(AtomicDataDict.ENV_CELL_SHIFT_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ENV_CELL_SHIFT_KEY] = _lmdb_tensor(
                    data_dict[AtomicDataDict.ENV_CELL_SHIFT_KEY], torch.get_default_dtype()
                )
            if data_dict.get(AtomicDataDict.ONSITENV_INDEX_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ONSITENV_INDEX_KEY] = _lmdb_tensor(
                    data_dict[AtomicDataDict.ONSITENV_INDEX_KEY], torch.long
                )
            if data_dict.get(AtomicDataDict.ONSITENV_CELL_SHIFT_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY] = _lmdb_tensor(
                    data_dict[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY], torch.get_default_dtype()
                )
            atomicdata = AtomicData(
                pos=_lmdb_tensor(pos.reshape(-1, 3), torch.get_default_dtype()),
                cell=_lmdb_tensor(cell.reshape(3, 3), torch.get_default_dtype()),
                atomic_numbers=_lmdb_tensor(atomic_numbers, torch.long),
                pbc=_lmdb_tensor(pbc, torch.bool),
                **atomicdata_kwargs,
            )
        else:
            atomicdata = AtomicData.from_points(
                pos=pos.reshape(-1, 3),
                cell=cell.reshape(3, 3),
                atomic_numbers=atomic_numbers,
                pbc=pbc,
                **info
            )
        self.info_files[self.file_map[idx]].update(cache_info)

        num_edges = atomicdata[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        num_nodes = atomicdata.num_nodes

        if has_pre_main and has_pre_overlap:
            pre_node_features = _expand_soc_uureal_compact(
                pre_node_features,
                data_dict,
                field_name=AtomicDataDict.NODE_FEATURES_KEY,
                keep_mask=soc_uureal_keep_mask,
            )
            pre_edge_features = _expand_soc_uureal_compact(
                pre_edge_features,
                data_dict,
                field_name=AtomicDataDict.EDGE_FEATURES_KEY,
                keep_mask=soc_uureal_keep_mask,
            )
            if pre_node_features.shape[0] != num_nodes or pre_edge_features.shape[0] != num_edges:
                raise ValueError(
                    "Precomputed LMDB feature rows do not match the active graph: "
                    f"node_features={tuple(pre_node_features.shape)}, "
                    f"edge_features={tuple(pre_edge_features.shape)}, "
                    f"num_nodes={num_nodes}, num_edges={num_edges}."
                )
            atomicdata[AtomicDataDict.NODE_FEATURES_KEY] = pre_node_features
            atomicdata[AtomicDataDict.EDGE_FEATURES_KEY] = pre_edge_features
            if need_overlap_features:
                pre_edge_overlap = torch.as_tensor(pre_edge_overlap)
                if pre_edge_overlap.shape[0] != num_edges:
                    raise ValueError(
                        "Precomputed LMDB edge overlap rows do not match the active graph: "
                        f"edge_overlap={tuple(pre_edge_overlap.shape)}, num_edges={num_edges}."
                    )
                if not self.orthogonal:
                    pre_node_overlap = torch.as_tensor(pre_node_overlap)
                    if pre_node_overlap.shape[0] != num_nodes:
                        raise ValueError(
                            "Precomputed LMDB node overlap rows do not match the active graph: "
                            f"node_overlap={tuple(pre_node_overlap.shape)}, num_nodes={num_nodes}."
                        )
                    atomicdata[AtomicDataDict.NODE_OVERLAP_KEY] = pre_node_overlap
                atomicdata[AtomicDataDict.EDGE_OVERLAP_KEY] = pre_edge_overlap
        elif self.get_Hamiltonian or self.get_DM or self.get_overlap:
            block_to_feature(atomicdata, self.type_mapper, blocks, overlap, self.orthogonal)

        if self.get_H0:
            if self.prefer_precomputed_h0 and node_h0 is not None and edge_h0 is not None:
                node_h0 = _expand_soc_uureal_compact(
                    node_h0,
                    data_dict,
                    field_name=AtomicDataDict.NODE_H0_KEY,
                    keep_mask=soc_uureal_keep_mask,
                )
                edge_h0 = _expand_soc_uureal_compact(
                    edge_h0,
                    data_dict,
                    field_name=AtomicDataDict.EDGE_H0_KEY,
                    keep_mask=soc_uureal_keep_mask,
                )
                if node_h0.shape[0] != num_nodes or edge_h0.shape[0] != num_edges:
                    raise ValueError(
                        "Precomputed LMDB H0 rows do not match the active graph: "
                        f"node_h0={tuple(node_h0.shape)}, edge_h0={tuple(edge_h0.shape)}, "
                        f"num_nodes={num_nodes}, num_edges={num_edges}."
                    )
                atomicdata[AtomicDataDict.NODE_H0_KEY] = node_h0
                atomicdata[AtomicDataDict.EDGE_H0_KEY] = edge_h0
            elif h0_blocks is not None:
                block_to_feature(
                    atomicdata,
                    self.type_mapper,
                    h0_blocks,
                    False,
                    self.orthogonal,
                    node_field=AtomicDataDict.NODE_H0_KEY,
                    edge_field=AtomicDataDict.EDGE_H0_KEY,
                )
            elif node_h0 is not None and edge_h0 is not None:
                atomicdata[AtomicDataDict.NODE_H0_KEY] = _expand_soc_uureal_compact(
                    node_h0,
                    data_dict,
                    field_name=AtomicDataDict.NODE_H0_KEY,
                    keep_mask=soc_uureal_keep_mask,
                )
                atomicdata[AtomicDataDict.EDGE_H0_KEY] = _expand_soc_uureal_compact(
                    edge_h0,
                    data_dict,
                    field_name=AtomicDataDict.EDGE_H0_KEY,
                    keep_mask=soc_uureal_keep_mask,
                )

        # Optional AO-block targets/H0 produced by the blockwise NexTHAM
        # conversion path. Keep this side channel independent of the feature
        # path so existing RME training remains unchanged.
        for blockwise_key in (
            AtomicDataDict.NODE_DELTA_HAMIL_BLOCKS_KEY,
            AtomicDataDict.EDGE_DELTA_HAMIL_BLOCKS_KEY,
            AtomicDataDict.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
            AtomicDataDict.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
            AtomicDataDict.NODE_H0_BLOCKS_KEY,
            AtomicDataDict.EDGE_H0_BLOCKS_KEY,
            AtomicDataDict.NODE_H0_BLOCK_SHAPE_KEY,
            AtomicDataDict.EDGE_H0_BLOCK_SHAPE_KEY,
        ):
            if blockwise_key in data_dict:
                atomicdata[blockwise_key] = torch.as_tensor(data_dict[blockwise_key])

        if (
            self.get_Hamiltonian
            and blocks is not False
            and blocks is not None
            and AtomicDataDict.NODE_DELTA_HAMIL_BLOCKS_KEY not in atomicdata
        ):
            from dptb.data.interfaces.blockwise_tensor import attach_block_tensors, block_dict_to_ordered_tensors

            start_id = 0 if "0_0_0_0_0" in blocks else 1
            target_blocks = block_dict_to_ordered_tensors(
                atomicdata,
                self.type_mapper,
                blocks,
                start_id=start_id,
                complete_edges=True,
                strict_complete_edges=False,
            )
            attach_block_tensors(atomicdata, target_blocks, prefix="delta_hamil")

        if (
            self.get_H0
            and h0_blocks is not None
            and AtomicDataDict.NODE_H0_BLOCKS_KEY not in atomicdata
        ):
            from dptb.data.interfaces.blockwise_tensor import attach_block_tensors, block_dict_to_ordered_tensors

            start_id = 0 if "0_0_0_0_0" in h0_blocks else 1
            target_h0_blocks = block_dict_to_ordered_tensors(
                atomicdata,
                self.type_mapper,
                h0_blocks,
                start_id=start_id,
                complete_edges=True,
                strict_complete_edges=False,
            )
            attach_block_tensors(atomicdata, target_h0_blocks, prefix="h0")

        return atomicdata

    def E3statistics(self, model: torch.nn.Module = None):

        if not self.get_Hamiltonian and not self.get_DM:
            return None

        if model is not None:
            if not isinstance(model.node_prediction_h, torch.nn.Module):
                return None

        assert self.transform is not None
        idp = model.embedding.idp
        has_soc = model.embedding.idp.has_soc

        e3h = E3Hamiltonian(basis=idp.basis, decompose=True, soc=has_soc)
        idp.get_irreps()

        # [FIX] Correctly count n_scalar for both SOC (0e+0o) and non-SOC (0e) cases.
        # Original code only counted the first type of scalar in sorted irreps.
        # sorted_irreps = idp.orbpair_irreps.sort()[0].simplify()
        # n_scalar = sorted_irreps[0].mul if sorted_irreps[0].ir.l == 0 else 0
        n_scalar = sum(mul for mul, (l, p) in idp.orbpair_irreps if l == 0)

        # init a count dict of atom species
        count_at = {}
        for at, tp in idp.chemical_symbol_to_type.items():
            count_at[tp] = 0

        count_bt = {}
        for bt, tp in idp.bond_to_type.items():
            count_bt[tp] = 0

        # calculate norm & mean
        node_norm_ave = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        node_square_ave = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        node_norm_std = torch.ones(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        node_scalar_ave = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)
        node_scalar_square_ave = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)
        node_scalar_std = torch.ones(len(idp.chemical_symbol_to_type), n_scalar)
        edge_norm_ave = torch.zeros(len(idp.bond_types), idp.orbpair_irreps.num_irreps)
        edge_square_ave = torch.zeros(len(idp.bond_types), idp.orbpair_irreps.num_irreps)
        edge_norm_std = torch.ones(len(idp.bond_types), idp.orbpair_irreps.num_irreps)
        edge_scalar_ave = torch.zeros(len(idp.bond_types), n_scalar)
        edge_scalar_square_ave = torch.zeros(len(idp.bond_types), n_scalar)
        edge_scalar_std = torch.ones(len(idp.bond_types), n_scalar)

        for idx in tqdm(range(self.len()), desc="Collecting E3 irreps statistics: "):
            with torch.no_grad():
                atomicdata = idp(self.get(idx=idx)).to_dict()
                if atomicdata[AtomicDataDict.EDGE_FEATURES_KEY].abs().sum() < 1e-7:
                    continue
                atomicdata = e3h(atomicdata)

                subcount_at = {}
                for at, tp in idp.chemical_symbol_to_type.items():
                    subcount_at[tp] = 0

                subcount_bt = {}
                for bt, tp in idp.bond_to_type.items():
                    subcount_bt[tp] = 0

                onsite_mask = idp.mask_to_nrme[atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten()]

                for at, tp in idp.chemical_symbol_to_type.items():
                    count_scalar = 0
                    at_mask = onsite_mask[atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten().eq(tp)]
                    n_at = at_mask.shape[0]

                    if n_at > 0:
                        at_onsite = atomicdata[AtomicDataDict.NODE_FEATURES_KEY][
                            atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten().eq(tp)]
                        for ir, s in enumerate(idp.orbpair_irreps.slices()):
                            sub_tensor = at_onsite[:, s]
                            if sub_tensor.shape[-1] == 1:
                                count_scalar += 1
                            norms = torch.norm(sub_tensor, p=2, dim=1)
                            # we do a running avg and var here
                            node_norm_ave[tp][ir] = (node_norm_ave[tp][ir] * count_at[tp] + norms.sum(dim=0)) / (
                                        count_at[tp] + n_at)
                            node_square_ave[tp][ir] = (node_square_ave[tp][ir] * count_at[tp] + (norms ** 2).sum(
                                dim=0)) / (count_at[tp] + n_at)
                            if count_at[tp] + n_at > 1:
                                node_norm_std[tp][ir] = torch.nan_to_num(torch.sqrt(
                                    (count_at[tp] + n_at) / (count_at[tp] + n_at - 1) * (
                                                node_square_ave[tp][ir] - node_norm_ave[tp][ir] ** 2)), nan=0.0)
                            else:
                                node_norm_std[tp][ir] = 1.0

                            if sub_tensor.shape[-1] == 1:
                                # is scalar
                                node_scalar_ave[tp][count_scalar - 1] = (node_scalar_ave[tp][count_scalar - 1] *
                                                                         count_at[tp] + sub_tensor.sum()) / (
                                                                                    count_at[tp] + n_at)
                                node_scalar_square_ave[tp][count_scalar - 1] = (node_scalar_square_ave[tp][
                                                                                    count_scalar - 1] * count_at[tp] + (
                                                                                            sub_tensor ** 2).sum()) / (
                                                                                           count_at[tp] + n_at)
                                if count_at[tp] + n_at > 1:
                                    node_scalar_std[tp][count_scalar - 1] = torch.nan_to_num(torch.sqrt(
                                        (count_at[tp] + n_at) / (count_at[tp] + n_at - 1) * (
                                                    node_scalar_square_ave[tp][count_scalar - 1] - node_scalar_ave[tp][
                                                count_scalar - 1] ** 2)), nan=0.0)
                                else:
                                    node_scalar_std[tp][count_scalar - 1] = 1.0
                        subcount_at[tp] = n_at
                        count_at[tp] += n_at
                assert sum(subcount_at.values()) == atomicdata[AtomicDataDict.POSITIONS_KEY].shape[0]

                # edge statistics
                hopping_mask = idp.mask_to_erme[atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten()]
                for bt, tp in idp.bond_to_type.items():
                    count_scalar = 0
                    bt_mask = hopping_mask[atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(tp)]
                    n_bt = bt_mask.shape[0]

                    if n_bt > 0:
                        bt_hopping = atomicdata[AtomicDataDict.EDGE_FEATURES_KEY][
                            atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(tp)]
                        for ir, s in enumerate(idp.orbpair_irreps.slices()):
                            sub_tensor = bt_hopping[:, s]
                            if sub_tensor.shape[-1] == 1:
                                count_scalar += 1

                            norms = torch.norm(sub_tensor, p=2, dim=1)
                            # we do a running avg and var here
                            edge_norm_ave[tp][ir] = (edge_norm_ave[tp][ir] * count_bt[tp] + norms.sum(dim=0)) / (
                                        count_bt[tp] + n_bt)
                            edge_square_ave[tp][ir] = (edge_square_ave[tp][ir] * count_bt[tp] + (norms ** 2).sum(
                                dim=0)) / (count_bt[tp] + n_bt)
                            if count_bt[tp] + n_bt > 1:
                                edge_norm_std[tp][ir] = torch.nan_to_num(torch.sqrt(
                                    (count_bt[tp] + n_bt) / (count_bt[tp] + n_bt - 1) * (
                                                edge_square_ave[tp][ir] - edge_norm_ave[tp][ir] ** 2)), nan=0.0)
                            else:
                                edge_norm_std[tp][ir] = 1.0
                            if sub_tensor.shape[-1] == 1:
                                # is scalar
                                edge_scalar_ave[tp][count_scalar - 1] = (edge_scalar_ave[tp][count_scalar - 1] *
                                                                         count_bt[tp] + sub_tensor.sum()) / (
                                                                                    count_bt[tp] + n_bt)
                                edge_scalar_square_ave[tp][count_scalar - 1] = (edge_scalar_square_ave[tp][
                                                                                    count_scalar - 1] * count_bt[tp] + (
                                                                                            sub_tensor ** 2).sum()) / (
                                                                                           count_bt[tp] + n_bt)
                                if count_bt[tp] + n_bt > 1:
                                    edge_scalar_std[tp][count_scalar - 1] = torch.nan_to_num(torch.sqrt(
                                        (count_bt[tp] + n_bt) / (count_bt[tp] + n_bt - 1) * (
                                                    edge_scalar_square_ave[tp][count_scalar - 1] - edge_scalar_ave[tp][
                                                count_scalar - 1] ** 2)), nan=0.0)
                                else:
                                    edge_scalar_std[tp][count_scalar - 1] = 1.0

                        subcount_bt[tp] = n_bt
                        count_bt[tp] += n_bt
                assert sum(subcount_bt.values()) == atomicdata[AtomicDataDict.EDGE_INDEX_KEY].shape[1]

        stats = {}
        stats["node"] = {
            "norm_ave": node_norm_ave,
            "norm_std": node_norm_std,
            "scalar_ave": node_scalar_ave,
            "scalar_std": node_scalar_std
        }
        stats["edge"] = {
            "norm_ave": edge_norm_ave,
            "norm_std": edge_norm_std,
            "scalar_ave": edge_scalar_ave,
            "scalar_std": edge_scalar_std,
        }

        if model is not None:
            # initilize the model param with statistics
            scalar_mask = torch.BoolTensor([ir.dim == 1 for ir in model.idp.orbpair_irreps])
            node_shifts = stats["node"]["scalar_ave"]
            node_scales = stats["node"]["norm_ave"]
            node_scales[:, scalar_mask] = stats["node"]["scalar_std"]

            edge_shifts = stats["edge"]["scalar_ave"]
            edge_scales = stats["edge"]["norm_ave"]
            edge_scales[:, scalar_mask] = stats["edge"]["scalar_std"]
            model.node_prediction_h.set_scale_shift(scales=node_scales, shifts=node_shifts)
            model.edge_prediction_h.set_scale_shift(scales=edge_scales, shifts=edge_shifts)

        return stats
