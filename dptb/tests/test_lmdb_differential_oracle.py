"""LMDB differential oracle: live record pipeline vs pre-refactor ``get()``.

PR-H (commit 90a48ba) decomposed the historical ~900-line ``LMDBDataset.get()``
into ``dptb/data/dataset/record_pipeline.py`` and promised byte-identical
behaviour.  This suite proves it differentially against a frozen vendor of the
pre-refactor implementation (``dptb/tests/_lmdb_oracle_85f87bb.py``, extracted
verbatim from commit 85f87bb -- the direct parent of the refactor commit).

For every lane in the flag matrix
``{plain H target, get_H0, full-H raw, prior residual} x {no prior, p2, p23}
x {AO blocks on/off}`` (plus Haar / offline-physical-H0 extensions) a live
``LMDBDataset`` and an ``OracleLMDBDataset`` are built over the *same* on-disk
LMDB record with independent (content-identical) OrbitalMappers, and
``get(idx)`` is compared exhaustively:

* identical AtomicData key sets;
* per key: identical dtype, shape, and tensor payload (``torch.equal`` plus
  raw ``numpy().tobytes()`` byte identity); non-tensors by ``==``;
* identical LMDB read telemetry and validated-contract cache state;
* a warm second ``get`` (validate-once-per-worker fast path) compared again.

Malformed records must raise the SAME exception type and message on both
implementations, twice in a row (fail-closed retries never poison state).

Record builders are reused from the sibling suites ``test_p2_prior_route``
(compact fingerprinted P2 / dual-prior P23 records) and
``test_nonsoc_cache_materialize`` (upper-triangle-mapper minimal P2 record).

A REAL divergence found by this suite is a successful outcome of the audit: it
must be characterised as an ``xfail`` here with a full explanation instead of
patching ``dptb/data``.  As of the current tree no divergence was observed.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
import pytest
import torch

# Sibling test modules (flat pytest layout: dptb/tests is on sys.path when
# tests import).  Their record builders are the fixtures this suite reuses.
import test_nonsoc_cache_materialize as _ncm
import test_p2_prior_route as _p2r
from _lmdb_oracle_85f87bb import OracleLMDBDataset

from dptb.data import _keys
from dptb.data.build import DatasetBuilder
from dptb.data.dataset.lmdb_dataset import LMDBDataset
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    BASIS_FINGERPRINT_KEY,
    P2_SAMPLE_SCHEMA,
    P2_SOURCE_FINGERPRINT_KEY,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
)
from dptb.data.transforms_upper_triangle import (
    OrbitalMapper as UpperTriangleOrbitalMapper,
)


# ===========================================================================
# Pairing helpers
# ===========================================================================


def _oracle_twin(dataset) -> OracleLMDBDataset:
    """Construct the frozen-oracle dataset with the live dataset's exact config.

    ``__init__`` is identical between the two classes (the refactor touched only
    ``get()``), so mirroring the resolved live attributes reproduces the same
    construction.  ``info_files`` entries are copied so the two datasets never
    share mutable metadata, and the donor's OrbitalMapper is reused unshared
    (the donor is discarded), so live and oracle never share lazy mapper state.
    """
    return OracleLMDBDataset(
        root=dataset.root,
        info_files={name: dict(entry) for name, entry in dataset.info_files.items()},
        type_mapper=dataset.type_mapper,
        orthogonal=dataset.orthogonal,
        get_Hamiltonian=dataset.get_Hamiltonian,
        get_H0=dataset.get_H0,
        get_prior=dataset.get_prior,
        residual_hamiltonian=dataset.residual_hamiltonian,
        get_overlap=dataset.get_overlap,
        get_DM=dataset.get_DM,
        get_eigenvalues=dataset.get_eigenvalues,
        h0_key=dataset.h0_key,
        prefer_precomputed_h0=dataset.prefer_precomputed_h0,
        prior_kind=dataset.prior_kind,
        prior_raw_key=dataset.prior_raw_key,
        prefer_precomputed_prior=dataset.prefer_precomputed_prior,
        require_full_h_target=dataset.require_full_h_target,
        expected_prior_source_fingerprint=dataset.expected_prior_source_fingerprint,
        audit_prior_representations=dataset.audit_prior_representations,
        require_prior_blocks=dataset.require_prior_blocks,
    )


def _builder_pair(root, **kwargs):
    """(live, oracle) datasets built twice from the same DatasetBuilder config.

    The second (donor) build supplies a fresh, content-identical OrbitalMapper
    and info_files to the oracle so no lazily-cached mapper state leaks from the
    live read order into the oracle read order or vice versa.
    """
    kwargs.setdefault("type", "LMDBDataset")
    kwargs.setdefault("separator", ".")
    kwargs.setdefault("basis", {"H": "1s"})
    kwargs.setdefault("r_max", 2.0)
    live = DatasetBuilder()(root=str(root), **kwargs)
    donor = DatasetBuilder()(root=str(root), **kwargs)
    assert isinstance(live, LMDBDataset)
    return live, _oracle_twin(donor)


_DIRECT_INFO = {
    "r_max": 2.0,
    "er_max": None,
    "oer_max": None,
    "wave_align": False,
    "train_w_homo_lumo_gap": False,
    "train_w_eps": False,
    "train_w_charge": False,
    "train_dip": False,
    "train_polar": False,
}


def _direct_pair(root, lmdb_folder, mapper_factory, **flags):
    """(live, oracle) constructed directly (bypassing DatasetBuilder)."""
    live = LMDBDataset(
        root=str(root),
        info_files={lmdb_folder: dict(_DIRECT_INFO)},
        type_mapper=mapper_factory(),
        **flags,
    )
    oracle = OracleLMDBDataset(
        root=str(root),
        info_files={lmdb_folder: dict(_DIRECT_INFO)},
        type_mapper=mapper_factory(),
        **flags,
    )
    return live, oracle


# ===========================================================================
# Exhaustive sample comparison
# ===========================================================================


def _describe(value) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(dtype={value.dtype}, shape={tuple(value.shape)})"
    if isinstance(value, np.ndarray):
        return f"ndarray(dtype={value.dtype}, shape={value.shape})"
    return f"{type(value).__name__}({value!r})"


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> str:
    try:
        if a.is_complex():
            return f"{(a - b).abs().max().item():.6e}"
        return f"{(a.double() - b.double()).abs().max().item():.6e}"
    except Exception:  # non-numeric dtypes: the raw report is enough
        return "n/a"


def _compare_tensor(key: str, a: torch.Tensor, b: torch.Tensor, problems: list) -> None:
    if a.dtype != b.dtype:
        problems.append(f"{key}: dtype live={a.dtype} vs oracle={b.dtype}")
        return
    if tuple(a.shape) != tuple(b.shape):
        problems.append(f"{key}: shape live={tuple(a.shape)} vs oracle={tuple(b.shape)}")
        return
    a_cpu = a.detach().cpu().contiguous()
    b_cpu = b.detach().cpu().contiguous()
    byte_identical = a_cpu.numpy().tobytes() == b_cpu.numpy().tobytes()
    if byte_identical:
        return
    if torch.equal(a_cpu, b_cpu):
        problems.append(
            f"{key}: torch.equal but NOT byte-identical "
            "(-0.0 vs +0.0 or differing NaN payloads)"
        )
    else:
        problems.append(
            f"{key}: tensor values differ, max|live-oracle|={_max_abs_diff(a_cpu, b_cpu)}"
        )


def _compare_samples(lane: str, live_sample, oracle_sample) -> None:
    problems: list = []
    live_dict = live_sample.to_dict()
    oracle_dict = oracle_sample.to_dict()
    only_live = sorted(set(live_dict) - set(oracle_dict))
    only_oracle = sorted(set(oracle_dict) - set(live_dict))
    if only_live:
        problems.append(f"keys only in live sample: {only_live}")
    if only_oracle:
        problems.append(f"keys only in oracle sample: {only_oracle}")
    for key in sorted(set(live_dict) & set(oracle_dict)):
        a, b = live_dict[key], oracle_dict[key]
        a_is_tensor = isinstance(a, torch.Tensor)
        b_is_tensor = isinstance(b, torch.Tensor)
        if a_is_tensor != b_is_tensor:
            problems.append(
                f"{key}: container mismatch live={_describe(a)} vs oracle={_describe(b)}"
            )
        elif a_is_tensor:
            _compare_tensor(key, a, b, problems)
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
                problems.append(
                    f"{key}: container mismatch live={_describe(a)} vs oracle={_describe(b)}"
                )
            elif a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                problems.append(
                    f"{key}: ndarray mismatch live={_describe(a)} vs oracle={_describe(b)}"
                )
        else:
            if type(a) is not type(b) or a != b:
                problems.append(
                    f"{key}: value mismatch live={_describe(a)} vs oracle={_describe(b)}"
                )
    assert not problems, (
        f"[{lane}] live record pipeline diverges from the 85f87bb oracle:\n  "
        + "\n  ".join(problems)
    )


def _compare_read_telemetry(lane: str, live, oracle) -> None:
    assert live._last_lmdb_pickle_bytes == oracle._last_lmdb_pickle_bytes, lane
    assert live._last_lmdb_record_identity == oracle._last_lmdb_record_identity, lane


def _compare_contract_caches(lane: str, live, oracle) -> None:
    live_cache = live._validated_record_contracts
    oracle_cache = oracle._validated_record_contracts
    assert set(live_cache) == set(oracle_cache), (
        f"[{lane}] validated-contract cache keys differ: "
        f"live={sorted(map(str, live_cache))} oracle={sorted(map(str, oracle_cache))}"
    )
    for cache_key in live_cache:
        for slot, (a, b) in enumerate(zip(live_cache[cache_key], oracle_cache[cache_key])):
            assert np.array_equal(np.asarray(a), np.asarray(b)), (
                f"[{lane}] cached canonical graph slot {slot} differs for {cache_key}"
            )


def _run_pair(lane: str, live, oracle, idx: int = 0):
    """Cold + warm differential comparison for one lane."""
    assert live.len() == oracle.len()
    live_cold = live.get(idx)
    oracle_cold = oracle.get(idx)
    _compare_samples(f"{lane}/cold", live_cold, oracle_cold)
    _compare_read_telemetry(f"{lane}/cold", live, oracle)
    _compare_contract_caches(f"{lane}/cold", live, oracle)
    # Warm read: the validate-once-per-worker fast path (cached canonical graph,
    # check_finite=False re-validation skips) must also be byte-identical.
    live_warm = live.get(idx)
    oracle_warm = oracle.get(idx)
    _compare_samples(f"{lane}/warm", live_warm, oracle_warm)
    _compare_contract_caches(f"{lane}/warm", live, oracle)
    return live_cold


def _assert_same_error(lane: str, live, oracle, idx: int = 0, attempts: int = 2):
    """Both implementations raise the same exception type+message, repeatably."""
    observed = []
    for name, dataset in (("live", live), ("oracle", oracle)):
        per_dataset = []
        for _ in range(attempts):
            with pytest.raises(Exception) as excinfo:
                dataset.get(idx)
            per_dataset.append((type(excinfo.value), str(excinfo.value)))
        assert per_dataset[0] == per_dataset[-1], (
            f"[{lane}] {name} error is not stable across retries "
            "(failed read poisoned dataset state): "
            f"{per_dataset[0]!r} then {per_dataset[-1]!r}"
        )
        observed.append(per_dataset[0])
    (live_type, live_msg), (oracle_type, oracle_msg) = observed
    assert live_type is oracle_type, (
        f"[{lane}] exception type diverged: live={live_type.__name__} "
        f"({live_msg}) vs oracle={oracle_type.__name__} ({oracle_msg})"
    )
    assert live_msg == oracle_msg, (
        f"[{lane}] exception message diverged:\n  live  : {live_msg}\n  oracle: {oracle_msg}"
    )
    # A failed read must never mark the record trusted on either side.
    assert live._validated_record_contracts == {}
    assert oracle._validated_record_contracts == {}
    return live_type, live_msg


# ===========================================================================
# Record builders (raw historical records; compact ones come from _p2r/_ncm)
# ===========================================================================


def _raw_record(
    *,
    with_schema: bool = True,
    h0_scale: float = 0.95,
    bad_p2_key: bool = False,
    with_haar_physical: bool = False,
):
    """A raw NexTHam-style record: AO block dicts, stored graph, no fingerprints.

    Mirrors the record of test_p2_prior_route.test_lmdb_raw_full_h_plus_p2_
    contract_end_to_end, extended with an ``hamiltonian_0`` dict (H0 = scale*H,
    so H-H0 shrinks by 1/(1-scale) for the residual lanes), a raw ``overlap``
    dict, and optional Haar / offline-physical-H0 row-aligned extensions.
    """
    h_blocks = {
        "0_0_0_0_0": np.asarray([[1.2]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[1.4]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[0.3]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[0.3]], dtype=np.float32),
    }
    h0_blocks = {
        key: (value * h0_scale).astype(np.float32) for key, value in h_blocks.items()
    }
    p2_blocks = {
        "0_0_0_0_0": np.asarray([[1.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[1.1]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[0.2]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[0.2]], dtype=np.float32),
    }
    p23_blocks = {
        "0_0_0_0_0": np.asarray([[1.5]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[1.6]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[0.4]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[0.4]], dtype=np.float32),
    }
    if bad_p2_key:
        p2_blocks["bad-key"] = np.asarray([[0.5]], dtype=np.float32)
    overlap_blocks = {
        "0_0_0_0_0": np.asarray([[1.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[1.0]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[0.1]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[0.1]], dtype=np.float32),
    }
    record = {
        _keys.CELL_KEY: np.eye(3, dtype=np.float32) * 8.0,
        _keys.POSITIONS_KEY: np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1], dtype=np.int64),
        _keys.PBC_KEY: np.asarray([False, False, False]),
        _keys.EDGE_INDEX_KEY: np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        _keys.EDGE_CELL_SHIFT_KEY: np.zeros((2, 3), dtype=np.float32),
        "hamiltonian": h_blocks,
        "hamiltonian_0": h0_blocks,
        "hamiltonian_p2": p2_blocks,
        "hamiltonian_p23": p23_blocks,
        "overlap": overlap_blocks,
    }
    if with_schema:
        record.update(
            {
                SAMPLE_SCHEMA_KEY: P2_SAMPLE_SCHEMA,
                TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS,
                TARGET_SOURCE_KEY: "raw_hamiltonian",
            }
        )
    if with_haar_physical:
        # Row-aligned to the stored 2-node / 2-edge graph.  No schema key on
        # this variant: historical records carry these fields without the v2
        # row-aligned fingerprint contract.
        record.update(
            {
                _keys.HAAR_U0_KEY: np.eye(2, dtype=np.float32),
                _keys.HAAR_NODE_FEATURES_KEY: np.arange(8, dtype=np.float32).reshape(2, 4),
                _keys.HAAR_EDGE_FEATURES_KEY: (
                    np.arange(8, dtype=np.float32).reshape(2, 4) + 0.5
                ),
                _keys.NODE_PHYSICAL_H0_KEY: (
                    np.arange(8, dtype=np.float32).reshape(2, 4) * 0.1
                ),
                _keys.EDGE_PHYSICAL_H0_KEY: (
                    np.arange(8, dtype=np.float32).reshape(2, 4) * 0.2 - 0.3
                ),
            }
        )
    return record


# ===========================================================================
# Healthy lanes: raw records
# ===========================================================================

_RAW_LANES = [
    # (lane id, record kwargs, dataset kwargs)
    ("raw-plain-h-no-prior", {}, dict(get_Hamiltonian=True)),
    (
        "raw-plain-h-overlap",
        {},
        dict(get_Hamiltonian=True, get_overlap=True),
    ),
    (
        "raw-h0-no-prior",
        {},
        dict(get_Hamiltonian=True, get_H0=True),
    ),
    (
        "raw-full-h-no-prior",
        {},
        dict(get_Hamiltonian=True, require_full_h_target=True),
    ),
    (
        "raw-residual-no-prior",
        {},
        dict(get_Hamiltonian=True, residual_hamiltonian=True),
    ),
    (
        "raw-full-h-p2-blocks-on",
        {},
        dict(
            get_Hamiltonian=True,
            get_P2=True,
            require_p2_blocks=True,
            require_full_h_target=True,
            audit_p2_representations=True,
        ),
    ),
    (
        "raw-p2-blocks-off",
        {},
        dict(get_Hamiltonian=True, get_prior=True, prior_kind="p2"),
    ),
    (
        "raw-residual-p2-h0",
        {},
        dict(
            get_Hamiltonian=True,
            residual_hamiltonian=True,
            get_H0=True,
            get_prior=True,
            prior_kind="p2",
        ),
    ),
    (
        "raw-full-h-p23-blocks-on",
        {},
        dict(
            get_Hamiltonian=True,
            get_prior=True,
            prior_kind="p23",
            require_prior_blocks=True,
            require_full_h_target=True,
            audit_prior_representations=True,
        ),
    ),
    (
        "raw-residual-p23-blocks-off",
        {},
        dict(
            get_Hamiltonian=True,
            residual_hamiltonian=True,
            get_prior=True,
            prior_kind="p23",
        ),
    ),
    (
        "raw-haar-physical-h0",
        dict(with_schema=False, with_haar_physical=True),
        dict(get_Hamiltonian=True),
    ),
    (
        "raw-haar-physical-h0-with-h0-and-p2",
        dict(with_schema=False, with_haar_physical=True),
        dict(get_Hamiltonian=True, get_H0=True, get_prior=True, prior_kind="p2"),
    ),
]


@pytest.mark.parametrize(
    "lane,record_kwargs,dataset_kwargs",
    _RAW_LANES,
    ids=[lane for lane, _, _ in _RAW_LANES],
)
def test_raw_record_lanes_match_oracle(tmp_path, lane, record_kwargs, dataset_kwargs):
    _p2r._write_single_lmdb(tmp_path, "oracle-raw", _raw_record(**record_kwargs))
    live, oracle = _builder_pair(tmp_path, prefix="oracle-raw", **dataset_kwargs)
    _run_pair(lane, live, oracle)


def test_raw_residual_target_actually_shrinks_and_matches(tmp_path):
    """The residual lane really exercises H-H0 (delta != H) on both sides."""
    _p2r._write_single_lmdb(tmp_path, "oracle-raw", _raw_record())
    live, oracle = _builder_pair(
        tmp_path,
        prefix="oracle-raw",
        get_Hamiltonian=True,
        residual_hamiltonian=True,
    )
    sample = _run_pair("raw-residual-content", live, oracle)
    node_delta = sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY]
    # H0 = 0.95*H, so the residual is 0.05*H: onsite entries 0.06 / 0.07.
    torch.testing.assert_close(
        node_delta.flatten(),
        torch.tensor([1.2 * 0.05, 1.4 * 0.05]),
        atol=1.0e-6,
        rtol=0.0,
    )


# ===========================================================================
# Healthy lanes: compact fingerprinted P2 records (builders from _p2r)
# ===========================================================================


def test_compact_p2_blocks_on_matches_oracle(tmp_path):
    record, p2_source = _p2r._compact_p2_record()
    _p2r._write_single_lmdb(tmp_path, "oracle-compact", record)
    live = _p2r._compact_dataset(tmp_path, "oracle-compact", p2_source)
    oracle = _oracle_twin(_p2r._compact_dataset(tmp_path, "oracle-compact", p2_source))
    _run_pair("compact-p2-blocks-on", live, oracle)


def test_compact_p2_blocks_off_matches_oracle(tmp_path):
    record, p2_source = _p2r._compact_p2_record()
    _p2r._write_single_lmdb(tmp_path, "oracle-compact", record)
    live = _p2r._compact_selected_prior_dataset(
        tmp_path, "oracle-compact", p2_source, prior_kind="p2", require_blocks=False
    )
    oracle = _oracle_twin(
        _p2r._compact_selected_prior_dataset(
            tmp_path, "oracle-compact", p2_source, prior_kind="p2", require_blocks=False
        )
    )
    _run_pair("compact-p2-blocks-off", live, oracle)


def test_compact_p2_prefer_precomputed_off_matches_oracle(tmp_path):
    """Third prior branch: RME features accepted without the prefer switch."""
    record, p2_source = _p2r._compact_p2_record()
    _p2r._write_single_lmdb(tmp_path, "oracle-compact", record)
    config = dict(
        er_max=3.0,
        get_Hamiltonian=True,
        get_prior=True,
        prior_kind="p2",
        prefer_precomputed_prior=False,
        require_prior_blocks=False,
        require_full_h_target=True,
        audit_prior_representations=False,
        expected_prior_source_fingerprint=p2_source,
        residual_hamiltonian=False,
    )
    live, oracle = _builder_pair(tmp_path, prefix="oracle-compact", **config)
    _run_pair("compact-p2-prefer-off", live, oracle)


def test_compact_no_prior_row_aligned_matches_oracle(tmp_path):
    """Schema-v2 record read with the prior disabled: precomputed main features
    plus the row-aligned graph fingerprint contract, no prior fields."""
    record, _ = _p2r._compact_p2_record()
    _p2r._write_single_lmdb(tmp_path, "oracle-compact", record)
    live, oracle = _builder_pair(
        tmp_path, prefix="oracle-compact", get_Hamiltonian=True
    )
    sample = _run_pair("compact-no-prior", live, oracle)
    assert _keys.NODE_P2_KEY not in sample
    assert _keys.NODE_P2_BLOCKS_KEY not in sample


# ===========================================================================
# Healthy lanes: dual-prior (P2+P23) records (builders from _p2r)
# ===========================================================================


@pytest.mark.parametrize(
    "lane,prior_kind,require_blocks,audit",
    [
        ("dual-p23-blocks-on", "p23", True, True),
        ("dual-p23-blocks-off", "p23", False, False),
        ("dual-p2-view-blocks-off", "p2", False, False),
    ],
    ids=["p23-blocks-on", "p23-blocks-off", "p2-view"],
)
def test_dual_prior_lanes_match_oracle(tmp_path, lane, prior_kind, require_blocks, audit):
    record, p2_source, p23_source = _p2r._compact_dual_prior_record()
    _p2r._write_single_lmdb(tmp_path, "oracle-dual", record)
    source = p23_source if prior_kind == "p23" else p2_source

    def _build():
        return _p2r._compact_selected_prior_dataset(
            tmp_path,
            "oracle-dual",
            source,
            prior_kind=prior_kind,
            require_blocks=require_blocks,
            audit=audit,
        )

    live = _build()
    oracle = _oracle_twin(_build())
    sample = _run_pair(lane, live, oracle)
    if prior_kind == "p23":
        assert _keys.NODE_P23_KEY in sample
        assert _keys.NODE_P2_KEY not in sample
    else:
        assert _keys.NODE_P2_KEY in sample
        assert _keys.NODE_P23_KEY not in sample


def test_dual_prior_plain_h_target_matches_oracle(tmp_path):
    """{plain H target} x {p23}: the dual record consumed WITHOUT the absolute
    full-H contract (require_full_h_target=False)."""
    record, _, p23_source = _p2r._compact_dual_prior_record()
    _p2r._write_single_lmdb(tmp_path, "oracle-dual", record)
    live, oracle = _builder_pair(
        tmp_path,
        prefix="oracle-dual",
        get_Hamiltonian=True,
        get_prior=True,
        prior_kind="p23",
        expected_prior_source_fingerprint=p23_source,
    )
    sample = _run_pair("dual-p23-plain-h", live, oracle)
    assert _keys.NODE_P23_KEY in sample


def _dual_record_with_precomputed_h0():
    record, _, p23_source = _p2r._compact_dual_prior_record()
    record[_keys.NODE_H0_KEY] = np.asarray([[0.9], [1.0]], dtype=np.float32)
    record[_keys.EDGE_H0_KEY] = np.asarray([[0.15], [0.15]], dtype=np.float32)
    # node_h0/edge_h0 are row-aligned candidates: re-seal the row fingerprints.
    _p2r._refresh_dual_prior_fingerprints(record)
    return record, p23_source


@pytest.mark.parametrize(
    "lane,prefer_precomputed_h0",
    [("dual-p23-h0-precomputed", True), ("dual-p23-h0-prefer-off", False)],
    ids=["precomputed", "prefer-off"],
)
def test_dual_prior_h0_lanes_match_oracle(tmp_path, lane, prefer_precomputed_h0):
    """{get_H0} x {p23}: precomputed node_h0/edge_h0 rows on the dual record.

    With prefer_precomputed_h0=False and no raw hamiltonian_0 dict the loader
    must fall through to the third H0 branch (direct row assignment without the
    row-count validation) -- on both implementations identically.
    """
    record, p23_source = _dual_record_with_precomputed_h0()
    _p2r._write_single_lmdb(tmp_path, "oracle-dual-h0", record)
    live, oracle = _builder_pair(
        tmp_path,
        prefix="oracle-dual-h0",
        get_Hamiltonian=True,
        get_H0=True,
        prefer_precomputed_h0=prefer_precomputed_h0,
        get_prior=True,
        prior_kind="p23",
        expected_prior_source_fingerprint=p23_source,
    )
    sample = _run_pair(lane, live, oracle)
    torch.testing.assert_close(
        sample[_keys.NODE_H0_KEY], torch.tensor([[0.9], [1.0]])
    )


# ===========================================================================
# Healthy lane: upper-triangle-mapper record (builder from _ncm)
# ===========================================================================


def _ncm_mapper():
    return UpperTriangleOrbitalMapper({"H": ["1s"]}, method="e3tb", device="cpu")


def test_upper_triangle_minimal_p2_record_matches_oracle(tmp_path):
    """_ncm's minimal schema-v2 P2 record, completed with a source fingerprint
    and refreshed bundle fingerprints, read through directly-constructed
    datasets that use the upper-triangle OrbitalMapper."""
    record = _ncm._minimal_p2_record(_ncm_mapper())
    p2_source = hashlib.sha256(b"oracle-differential-ncm-p2-source").hexdigest()
    record[P2_SOURCE_FINGERPRINT_KEY] = p2_source
    _p2r._refresh_compact_fingerprints(record)
    _p2r._write_single_lmdb(tmp_path, "oracle-ncm", record)

    live, oracle = _direct_pair(
        tmp_path,
        "oracle-ncm.lmdb",
        _ncm_mapper,
        get_prior=True,
        prior_kind="p2",
        expected_prior_source_fingerprint=p2_source,
    )
    sample = _run_pair("ncm-upper-triangle-p2", live, oracle)
    torch.testing.assert_close(
        sample[_keys.NODE_P2_KEY], torch.tensor([[1.0], [2.0]])
    )
    # AO blocks stay off this lane (require/audit disabled).
    assert _keys.NODE_P2_BLOCKS_KEY not in sample


# ===========================================================================
# Malformed records: identical exception type + message on both sides
# ===========================================================================


def _tampered_row_fingerprint_record():
    record, p2_source = _p2r._compact_p2_record()
    record[_keys.NODE_P2_KEY] = record[_keys.NODE_P2_KEY].copy()
    record[_keys.NODE_P2_KEY][0, 0] += 0.25  # NOT refreshed: fingerprint mismatch
    return record, p2_source


def _wrong_shape_node_p2_record():
    record, p2_source = _p2r._compact_p2_record()
    # Three rows against a two-node graph; fingerprints refreshed so the shape
    # validator (not the fingerprint gate) is what fails.
    record[_keys.NODE_P2_KEY] = np.asarray([[1.0], [1.1], [1.2]], dtype=np.float32)
    _p2r._refresh_compact_fingerprints(record)
    return record, p2_source


def _partial_prior_record():
    record, p2_source = _p2r._compact_p2_record()
    # No fingerprint refresh: the XOR partial-prior gate fires while the sample
    # context is parsed, before any fingerprint is validated (and the refresh
    # helper itself requires both RME fields to be present).
    del record[_keys.EDGE_P2_KEY]
    return record, p2_source


def _tampered_basis_fingerprint_record():
    record, p2_source = _p2r._compact_p2_record()
    record[BASIS_FINGERPRINT_KEY] = hashlib.sha256(b"wrong-basis").hexdigest()
    return record, p2_source


def _partial_stored_graph_record():
    record, p2_source = _p2r._compact_p2_record()
    del record[_keys.EDGE_CELL_SHIFT_KEY]
    return record, p2_source


def _compact_pair(tmp_path, record, p2_source, *, require_blocks):
    _p2r._write_single_lmdb(tmp_path, "oracle-malformed", record)

    def _build():
        if require_blocks:
            return _p2r._compact_dataset(tmp_path, "oracle-malformed", p2_source)
        return _p2r._compact_selected_prior_dataset(
            tmp_path,
            "oracle-malformed",
            p2_source,
            prior_kind="p2",
            require_blocks=False,
        )

    return _build(), _oracle_twin(_build())


@pytest.mark.parametrize(
    "case,builder,require_blocks,expected_type,expected_fragment",
    [
        (
            "tampered-row-fingerprint",
            _tampered_row_fingerprint_record,
            False,
            ValueError,
            "row_aligned_data_fingerprint mismatch",
        ),
        (
            "wrong-shape-node-p2-rows",
            _wrong_shape_node_p2_record,
            False,
            ValueError,
            "node rows 3 do not match num_nodes=2",
        ),
        (
            "partial-prior-missing-edge-rme",
            _partial_prior_record,
            False,
            ValueError,
            "refusing a partial prior",
        ),
        (
            "tampered-basis-fingerprint",
            _tampered_basis_fingerprint_record,
            True,
            ValueError,
            "basis_fingerprint mismatch",
        ),
        (
            "partial-stored-graph",
            _partial_stored_graph_record,
            True,
            ValueError,
            "both edge_index and edge_cell_shift",
        ),
    ],
    ids=[
        "tampered-row-fingerprint",
        "wrong-shape-node-p2-rows",
        "partial-prior",
        "tampered-basis-fingerprint",
        "partial-stored-graph",
    ],
)
def test_malformed_compact_records_raise_identically(
    tmp_path, case, builder, require_blocks, expected_type, expected_fragment
):
    record, p2_source = builder()
    live, oracle = _compact_pair(tmp_path, record, p2_source, require_blocks=require_blocks)
    error_type, message = _assert_same_error(case, live, oracle)
    assert error_type is expected_type, (case, error_type, message)
    assert expected_fragment in message, (case, message)


def test_malformed_raw_invalid_ao_block_key_raises_identically(tmp_path):
    _p2r._write_single_lmdb(tmp_path, "oracle-raw-bad", _raw_record(bad_p2_key=True))
    live, oracle = _builder_pair(
        tmp_path, prefix="oracle-raw-bad", get_Hamiltonian=True, get_prior=True
    )
    error_type, message = _assert_same_error("invalid-ao-block-key", live, oracle)
    assert error_type is ValueError
    assert "invalid AO-block key" in message


def test_malformed_raw_residual_no_shrink_raises_identically(tmp_path):
    # H0 = 0.01*H leaves the "residual" at ~H scale: the delta-in-H-slot guard
    # must reject double subtraction with the same RuntimeError on both sides.
    _p2r._write_single_lmdb(
        tmp_path, "oracle-raw-noshrink", _raw_record(h0_scale=0.01)
    )
    live, oracle = _builder_pair(
        tmp_path,
        prefix="oracle-raw-noshrink",
        get_Hamiltonian=True,
        residual_hamiltonian=True,
    )
    error_type, message = _assert_same_error("residual-no-shrink", live, oracle)
    assert error_type is RuntimeError
    assert "does not shrink" in message
