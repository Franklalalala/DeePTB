"""Factorized local-three-centre VNA tables and graph assembly.

This module deliberately keeps storage-field names out of the physical
implementation.  A dual-prior training record retains its qualified P2 fields
and adds independent ``node_p23``/``edge_p23`` plus AO-block fields containing

    P23 = P2 + explicit factorized third-centre neutral-atom-potential blocks.

The expensive radial SBT work is performed by the table builder.  Structure
assembly below only enumerates periodic centres, interpolates/rotates reusable
two-centre factors and contracts small projector matrices.  Reverse graph rows
are derived from one representative block so real non-SOC Hermiticity is exact
instead of being accepted through a relaxed tolerance.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .p2_batch import VectorizedNearbyImageEnumerator
from .p2_table import RadialBlockTable


P23_VNA_TABLE_SCHEMA = "deeptb.p23_factorized_vna_table/v1"
P23_VNA_ASSEMBLY_SCHEMA = "deeptb.p23_factorized_vna_assembly/v1"
P23_FACTOR_SUPPORT_SEMANTICS = (
    "strict distance < cutoff(q=V*p) + cutoff(AO); "
    "q cutoff is min(physical VNA cutoff, PAO projector cutoff)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _factor_key(vna_species: str, ao_species: str) -> str:
    return f"{vna_species}|{ao_species}"


class P23VNAFactorTableStore:
    """Lazy checksum-aware reader for factorized VNA3c radial tables."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_cached_tables: int = 96,
        verify_checksums: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest_path = manifest_path
        self.manifest_sha256 = _sha256(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != P23_VNA_TABLE_SCHEMA:
            raise ValueError(
                f"P23 VNA table schema {self.manifest.get('schema')!r} != "
                f"{P23_VNA_TABLE_SCHEMA!r}"
            )
        if self.manifest.get("complete") is not True:
            raise ValueError("P23 VNA table manifest is not complete")
        if self.manifest.get("length_unit") != "bohr":
            raise ValueError("P23 VNA table length unit must be bohr")
        if self.manifest.get("factor_energy_unit") != "eV":
            raise ValueError("P23 factor tables must store eV-valued overlaps")
        if self.manifest.get("epsilon_unit") != "1/eV":
            raise ValueError("P23 projector epsilon unit must be 1/eV")
        if self.manifest.get("harmonic_convention") != "deeptb_abacus_real":
            raise ValueError("P23 tables must use the explicit DeePTB/ABACUS real gauge")
        if self.manifest.get("endpoint_policy") != "exclude_i0_and_jR":
            raise ValueError("P23 endpoint policy is not the P2-compatible contract")
        if self.manifest.get("factor_support_semantics") != (
            P23_FACTOR_SUPPORT_SEMANTICS
        ):
            raise ValueError("P23 factor support semantics are missing or incompatible")

        self.species: Mapping[str, Mapping[str, Any]] = self.manifest["species"]
        self.factor_tables: Mapping[str, Mapping[str, Any]] = self.manifest[
            "factor_tables"
        ]
        self.max_cached_tables = max(1, int(max_cached_tables))
        self.verify_checksums = bool(verify_checksums)
        self._table_cache: OrderedDict[str, RadialBlockTable] = OrderedDict()
        self._species_arrays: dict[str, dict[str, np.ndarray]] = {}
        self._verified: set[Path] = set()

        for symbol, row in self.species.items():
            shells = tuple(int(value) for value in row["orbital_shells"])
            q_shells = tuple(int(value) for value in row["vna_projector_shells"])
            if int(row["orbital_norb"]) != sum(2 * l + 1 for l in shells):
                raise ValueError(f"{symbol}: orbital shell/dimension mismatch")
            if int(row["vna_projector_norb"]) != sum(
                2 * l + 1 for l in q_shells
            ):
                raise ValueError(f"{symbol}: VNA-projector shell/dimension mismatch")
            if float(row["orbital_cutoff_bohr"]) <= 0.0:
                raise ValueError(f"{symbol}: invalid AO cutoff")
            if float(row["vna_cutoff_bohr"]) <= 0.0:
                raise ValueError(f"{symbol}: invalid VNA cutoff")

        for key, row in self.factor_tables.items():
            if "|" not in key:
                raise ValueError(f"invalid factor-table key {key!r}")
            vna, ao = key.split("|", 1)
            if vna not in self.species or ao not in self.species:
                raise ValueError(f"factor table {key!r} references unknown species")
            if str(row.get("path", "")).startswith(("/", "\\")):
                raise ValueError(f"factor table {key!r} path must be relative")

    def _verify(self, path: Path, expected: Any, *, label: str) -> None:
        if not self.verify_checksums or path in self._verified:
            return
        expected_text = str(expected or "").lower()
        if len(expected_text) != 64:
            raise ValueError(f"{label} has no valid SHA256")
        actual = _sha256(path)
        if actual != expected_text:
            raise ValueError(
                f"{label} checksum mismatch: manifest={expected_text}, actual={actual}"
            )
        self._verified.add(path)

    def has_factor(self, vna_species: str, ao_species: str) -> bool:
        return _factor_key(vna_species, ao_species) in self.factor_tables

    def factor(self, vna_species: str, ao_species: str) -> RadialBlockTable:
        key = _factor_key(vna_species, ao_species)
        cached = self._table_cache.get(key)
        if cached is not None:
            self._table_cache.move_to_end(key)
            return cached
        if key not in self.factor_tables:
            raise KeyError(f"P23 factor table is missing active pair {key}")
        metadata = self.factor_tables[key]
        path = (self.root / str(metadata["path"])).resolve()
        if self.root not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        self._verify(path, metadata.get("sha256"), label=f"factor:{key}")
        with np.load(path, allow_pickle=False) as payload:
            table = RadialBlockTable(
                distances=np.asarray(payload["distances"], dtype=np.float64),
                values=np.asarray(payload["values"]),
                left_shells=tuple(int(x) for x in payload["left_shells"]),
                right_shells=tuple(int(x) for x in payload["right_shells"]),
                support_bohr=float(np.asarray(payload["support_bohr"])),
                interpolation=str(self.manifest.get("interpolation", "cubic")),
            )
        expected_left = tuple(
            int(x) for x in self.species[vna_species]["vna_projector_shells"]
        )
        expected_right = tuple(
            int(x) for x in self.species[ao_species]["orbital_shells"]
        )
        if table.left_shells != expected_left or table.right_shells != expected_right:
            raise ValueError(f"factor:{key} shell contract mismatch")
        self._table_cache[key] = table
        self._table_cache.move_to_end(key)
        while len(self._table_cache) > self.max_cached_tables:
            self._table_cache.popitem(last=False)
        return table

    def _arrays(self, symbol: str) -> dict[str, np.ndarray]:
        cached = self._species_arrays.get(symbol)
        if cached is not None:
            return cached
        if symbol not in self.species:
            raise KeyError(symbol)
        metadata = self.species[symbol]
        path = (self.root / str(metadata["array_path"])).resolve()
        if self.root not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        self._verify(path, metadata.get("array_sha256"), label=f"species:{symbol}")
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        epsilon = np.asarray(arrays.get("epsilon_ao"), dtype=np.float32)
        expected = int(metadata["vna_projector_norb"])
        if epsilon.shape != (expected,) or not np.isfinite(epsilon).all():
            raise ValueError(f"{symbol}: invalid epsilon_ao array {epsilon.shape}")
        arrays["epsilon_ao"] = epsilon
        self._species_arrays[symbol] = arrays
        return arrays

    def epsilon(self, symbol: str) -> np.ndarray:
        return self._arrays(symbol)["epsilon_ao"]


@dataclass
class _QueryBucket:
    displacement_quantum_bohr: float
    key_to_row: dict[tuple[int, int, int], int]
    displacements: list[np.ndarray]
    values: np.ndarray | None = None

    @classmethod
    def create(cls, displacement_quantum_bohr: float) -> "_QueryBucket":
        return cls(float(displacement_quantum_bohr), {}, [])

    def add(self, displacement: np.ndarray) -> int:
        vector = np.asarray(displacement, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid factor displacement {vector}")
        quantum = self.displacement_quantum_bohr
        key = tuple(int(x) for x in np.rint(vector / quantum).astype(np.int64))
        row = self.key_to_row.get(key)
        if row is not None:
            previous = self.displacements[row]
            if float(np.max(np.abs(previous - vector), initial=0.0)) > 2.1 * quantum:
                raise ValueError("factor displacement quantization collision")
            return row
        row = len(self.displacements)
        self.key_to_row[key] = row
        self.displacements.append(vector.copy())
        return row


@dataclass(frozen=True)
class _Block:
    block_id: int
    kind: str
    row: int
    reverse_row: int
    i: int
    j: int
    shift: tuple[int, int, int]
    rows: int
    cols: int


class P23VNAFactorAssembler:
    """Assemble explicit local VNA3c additions for one canonical graph."""

    def __init__(
        self,
        store: P23VNAFactorTableStore,
        *,
        displacement_quantum_bohr: float = 1.0e-10,
        factor_dtype: np.dtype[Any] | str = np.float32,
        contraction_chunk_terms: int = 2048,
    ) -> None:
        self.store = store
        self.displacement_quantum_bohr = float(displacement_quantum_bohr)
        if not math.isfinite(self.displacement_quantum_bohr) or (
            self.displacement_quantum_bohr <= 0.0
        ):
            raise ValueError("displacement quantum must be positive")
        self.factor_dtype = np.dtype(factor_dtype)
        if self.factor_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("factor dtype must be float32 or float64")
        self.contraction_chunk_terms = max(1, int(contraction_chunk_terms))

    @staticmethod
    def _integer_shifts(edge_shift: Any) -> np.ndarray:
        raw = np.asarray(edge_shift, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError("edge_cell_shift must be finite with shape [E,3]")
        rounded = np.rint(raw)
        if float(np.max(np.abs(raw - rounded), initial=0.0)) > 1.0e-6:
            raise ValueError("edge_cell_shift contains non-integer values")
        return rounded.astype(np.int64)

    @staticmethod
    def _reverse_rows(edge_index: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if shifts.shape != (edge_index.shape[1], 3):
            raise ValueError("edge_index/edge_cell_shift row mismatch")
        lookup: dict[tuple[int, int, int, int, int], int] = {}
        for row, ((i, j), shift) in enumerate(zip(edge_index.T, shifts)):
            key = (int(i), int(j), *(int(x) for x in shift))
            if key in lookup:
                raise ValueError(f"duplicate directed graph key {key}")
            lookup[key] = row
        reverse = np.empty(edge_index.shape[1], dtype=np.int64)
        for row, ((i, j), shift) in enumerate(zip(edge_index.T, shifts)):
            key = (int(j), int(i), *(-shift).tolist())
            if key not in lookup:
                raise ValueError(f"directed graph row {row} has no reverse {key}")
            reverse[row] = lookup[key]
        if not np.array_equal(reverse[reverse], np.arange(len(reverse))):
            raise ValueError("reverse-edge mapping is not an involution")
        return reverse

    def _blocks(
        self,
        *,
        symbols: Sequence[str],
        edge_index: np.ndarray,
        shifts: np.ndarray,
        reverse: np.ndarray,
        node_shapes: np.ndarray,
        edge_shapes: np.ndarray,
    ) -> list[_Block]:
        blocks: list[_Block] = []
        for atom, shape in enumerate(node_shapes):
            rows, cols = (int(shape[0]), int(shape[1]))
            expected = int(self.store.species[symbols[atom]]["orbital_norb"])
            if (rows, cols) != (expected, expected):
                raise ValueError(
                    f"node {atom} ({symbols[atom]}) shape {(rows, cols)} != "
                    f"{(expected, expected)}"
                )
            blocks.append(
                _Block(len(blocks), "node", atom, atom, atom, atom, (0, 0, 0), rows, cols)
            )
        for row, reverse_row in enumerate(reverse.tolist()):
            if row > int(reverse_row):
                continue
            if row == int(reverse_row):
                raise ValueError("main graph must not duplicate onsite rows as edges")
            i, j = (int(x) for x in edge_index[:, row])
            rows, cols = (int(x) for x in edge_shapes[row])
            rrows, rcols = (int(x) for x in edge_shapes[int(reverse_row)])
            expected = (
                int(self.store.species[symbols[i]]["orbital_norb"]),
                int(self.store.species[symbols[j]]["orbital_norb"]),
            )
            if (rows, cols) != expected or (rrows, rcols) != expected[::-1]:
                raise ValueError(f"edge {row}/{reverse_row} AO shape contract mismatch")
            blocks.append(
                _Block(
                    len(blocks),
                    "edge",
                    row,
                    int(reverse_row),
                    i,
                    j,
                    tuple(int(x) for x in shifts[row]),
                    rows,
                    cols,
                )
            )
        return blocks

    @staticmethod
    def _append_group(
        groups: dict[tuple[str, str, str], list[list[int]]],
        key: tuple[str, str, str],
        block_id: int,
        left_row: int,
        right_row: int,
    ) -> None:
        rows = groups.get(key)
        if rows is None:
            rows = [[], [], []]
            groups[key] = rows
        rows[0].append(int(block_id))
        rows[1].append(int(left_row))
        rows[2].append(int(right_row))

    def _evaluate_queries(
        self, buckets: Mapping[tuple[str, str], _QueryBucket]
    ) -> dict[str, Any]:
        total_bytes = 0
        for (vna, ao), bucket in buckets.items():
            if not bucket.displacements:
                raise RuntimeError(f"empty active factor bucket {(vna, ao)}")
            table = self.store.factor(vna, ao)
            displacements = np.asarray(bucket.displacements, dtype=np.float64)
            values = table.evaluate_many(displacements).astype(
                self.factor_dtype, copy=False
            )
            expected = (
                len(displacements),
                int(self.store.species[vna]["vna_projector_norb"]),
                int(self.store.species[ao]["orbital_norb"]),
            )
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(
                    f"factor {(vna, ao)} values {values.shape} != {expected}"
                )
            bucket.values = values
            total_bytes += int(values.nbytes)
        return {
            "factor_buckets": len(buckets),
            "unique_factor_queries": sum(len(item.displacements) for item in buckets.values()),
            "factor_cache_bytes": total_bytes,
        }

    def _contract_groups(
        self,
        *,
        groups: Mapping[tuple[str, str, str], list[list[int]]],
        buckets: Mapping[tuple[str, str], _QueryBucket],
        output: np.ndarray,
    ) -> dict[str, int]:
        chunks = 0
        terms = 0
        for (left_symbol, right_symbol, centre_symbol), raw in groups.items():
            block_ids = np.asarray(raw[0], dtype=np.int64)
            left_rows = np.asarray(raw[1], dtype=np.int64)
            right_rows = np.asarray(raw[2], dtype=np.int64)
            if not (
                block_ids.shape == left_rows.shape == right_rows.shape
                and block_ids.ndim == 1
            ):
                raise RuntimeError("P23 contraction group arrays are misaligned")
            if block_ids.size == 0:
                continue
            if np.any(np.diff(block_ids) < 0):
                order = np.argsort(block_ids, kind="stable")
                block_ids = block_ids[order]
                left_rows = left_rows[order]
                right_rows = right_rows[order]
            left_values = buckets[(centre_symbol, left_symbol)].values
            right_values = buckets[(centre_symbol, right_symbol)].values
            if left_values is None or right_values is None:
                raise RuntimeError("factor queries were not evaluated")
            epsilon = self.store.epsilon(centre_symbol).astype(
                self.factor_dtype, copy=False
            )
            rows = int(self.store.species[left_symbol]["orbital_norb"])
            cols = int(self.store.species[right_symbol]["orbital_norb"])

            boundaries = np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.flatnonzero(np.diff(block_ids)) + 1,
                    np.asarray([len(block_ids)], dtype=np.int64),
                )
            )
            segment = 0
            while segment < len(boundaries) - 1:
                begin = int(boundaries[segment])
                end = int(boundaries[segment + 1])
                if end - begin > self.contraction_chunk_terms:
                    target = int(block_ids[begin])
                    for start in range(begin, end, self.contraction_chunk_terms):
                        stop = min(start + self.contraction_chunk_terms, end)
                        left = left_values[left_rows[start:stop]]
                        right = right_values[right_rows[start:stop]]
                        contribution = np.matmul(
                            (left * epsilon[None, :, None]).transpose(0, 2, 1),
                            right,
                        )
                        output[target, :rows, :cols] += np.sum(
                            contribution, axis=0, dtype=np.float64
                        )
                        chunks += 1
                    terms += end - begin
                    segment += 1
                    continue

                stop_segment = segment + 1
                while stop_segment < len(boundaries) - 1:
                    candidate_end = int(boundaries[stop_segment + 1])
                    if candidate_end - begin > self.contraction_chunk_terms:
                        break
                    stop_segment += 1
                stop = int(boundaries[stop_segment])
                left = left_values[left_rows[begin:stop]]
                right = right_values[right_rows[begin:stop]]
                contribution = np.matmul(
                    (left * epsilon[None, :, None]).transpose(0, 2, 1), right
                )
                local_ids = block_ids[begin:stop]
                starts = np.concatenate(
                    (
                        np.asarray([0], dtype=np.int64),
                        np.flatnonzero(np.diff(local_ids)) + 1,
                    )
                )
                sums = np.add.reduceat(
                    contribution.astype(np.float64, copy=False), starts, axis=0
                )
                unique_ids = local_ids[starts]
                output[unique_ids, :rows, :cols] += sums
                chunks += 1
                terms += stop - begin
                segment = stop_segment
        return {"contraction_groups": len(groups), "contraction_chunks": chunks, "terms": terms}

    def assemble_graph_addition(
        self,
        *,
        symbols: Sequence[str],
        positions_bohr: np.ndarray,
        cell_bohr: np.ndarray,
        edge_index: np.ndarray,
        edge_cell_shift: np.ndarray,
        node_shapes: np.ndarray,
        edge_shapes: np.ndarray,
        node_pad_shape: tuple[int, int],
        edge_pad_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        symbols = tuple(str(value) for value in symbols)
        positions = np.asarray(positions_bohr, dtype=np.float64)
        cell = np.asarray(cell_bohr, dtype=np.float64)
        edges = np.asarray(edge_index, dtype=np.int64)
        shifts = self._integer_shifts(edge_cell_shift)
        node_shapes = np.asarray(node_shapes, dtype=np.int64)
        edge_shapes = np.asarray(edge_shapes, dtype=np.int64)
        if positions.shape != (len(symbols), 3) or cell.shape != (3, 3):
            raise ValueError("invalid structure arrays for P23 assembly")
        if node_shapes.shape != (len(symbols), 2):
            raise ValueError("node block shapes do not match atom count")
        if edge_shapes.shape != (edges.shape[1], 2):
            raise ValueError("edge block shapes do not match graph")
        for symbol in symbols:
            if symbol not in self.store.species:
                raise KeyError(f"P23 table does not contain species {symbol}")

        reverse = self._reverse_rows(edges, shifts)
        blocks = self._blocks(
            symbols=symbols,
            edge_index=edges,
            shifts=shifts,
            reverse=reverse,
            node_shapes=node_shapes,
            edge_shapes=edge_shapes,
        )
        max_rows = max(int(node_pad_shape[0]), int(edge_pad_shape[0]))
        max_cols = max(int(node_pad_shape[1]), int(edge_pad_shape[1]))
        output = np.zeros((len(blocks), max_rows, max_cols), dtype=np.float64)
        buckets: dict[tuple[str, str], _QueryBucket] = {}
        groups: dict[tuple[str, str, str], list[list[int]]] = {}
        enumerator = VectorizedNearbyImageEnumerator(cell)
        zero = np.zeros(3, dtype=np.int64)
        onsite_terms = 0
        hopping_representative_terms = 0

        def query(vna: str, ao: str, displacement: np.ndarray) -> int:
            pair = (vna, ao)
            if not self.store.has_factor(vna, ao):
                raise KeyError(f"active P23 factor pair is absent: {vna}|{ao}")
            bucket = buckets.get(pair)
            if bucket is None:
                bucket = _QueryBucket.create(self.displacement_quantum_bohr)
                buckets[pair] = bucket
            return bucket.add(displacement)

        for block in blocks:
            i, j = block.i, block.j
            si, sj = symbols[i], symbols[j]
            centre_i = positions[i]
            shift = np.asarray(block.shift, dtype=np.int64)
            centre_j = positions[j] + shift @ cell
            edge_distance = float(np.linalg.norm(centre_j - centre_i))
            midpoint = 0.5 * (centre_i + centre_j)
            cutoff_i = float(self.store.species[si]["orbital_cutoff_bohr"])
            cutoff_j = float(self.store.species[sj]["orbital_cutoff_bohr"])
            block_terms = 0
            for k, sk in enumerate(symbols):
                support_k = float(self.store.species[sk]["vna_cutoff_bohr"])
                radius = support_k + max(cutoff_i, cutoff_j) + 0.5 * edge_distance
                translations, centres = enumerator.query_arrays(
                    positions[k], midpoint, radius
                )
                if len(centres) == 0:
                    continue
                left_delta = centre_i[None, :] - centres
                right_delta = centre_j[None, :] - centres
                active = (
                    np.linalg.norm(left_delta, axis=1)
                    < cutoff_i + support_k - 1.0e-12
                ) & (
                    np.linalg.norm(right_delta, axis=1)
                    < cutoff_j + support_k - 1.0e-12
                )
                active &= ~((k == i) & np.all(translations == zero[None, :], axis=1))
                active &= ~((k == j) & np.all(translations == shift[None, :], axis=1))
                for index in np.flatnonzero(active):
                    left_row = query(sk, si, left_delta[index])
                    right_row = query(sk, sj, right_delta[index])
                    self._append_group(
                        groups,
                        (si, sj, sk),
                        block.block_id,
                        left_row,
                        right_row,
                    )
                    block_terms += 1
            if block.kind == "node":
                onsite_terms += block_terms
            else:
                hopping_representative_terms += block_terms

        query_stats = self._evaluate_queries(buckets)
        contraction_stats = self._contract_groups(
            groups=groups, buckets=buckets, output=output
        )
        representative_terms = onsite_terms + hopping_representative_terms
        if contraction_stats["terms"] != representative_terms:
            raise RuntimeError("enumerated and contracted P23 term counts disagree")

        node = np.zeros((len(symbols), *node_pad_shape), dtype=np.float32)
        edge = np.zeros((edges.shape[1], *edge_pad_shape), dtype=np.float32)
        for block in blocks:
            value = output[block.block_id, : block.rows, : block.cols]
            if block.kind == "node":
                value = 0.5 * (value + value.T)
                node[block.row, : block.rows, : block.cols] = value.astype(np.float32)
            else:
                cast = value.astype(np.float32)
                edge[block.row, : block.rows, : block.cols] = cast
                edge[
                    block.reverse_row, : block.cols, : block.rows
                ] = cast.T.copy()
        if not np.isfinite(node).all() or not np.isfinite(edge).all():
            raise ValueError("assembled P23 VNA3c blocks contain NaN or infinity")

        directed_terms = onsite_terms + 2 * hopping_representative_terms
        stats: dict[str, Any] = {
            "schema": P23_VNA_ASSEMBLY_SCHEMA,
            "prior_kind": "P23_factorized_VNA3c",
            "table_manifest_sha256": self.store.manifest_sha256,
            "atoms": len(symbols),
            "directed_edges": int(edges.shape[1]),
            "representative_blocks": len(blocks),
            "onsite_terms": int(onsite_terms),
            "hopping_representative_terms": int(hopping_representative_terms),
            "directed_true_third_centre_terms": int(directed_terms),
            "factor_dtype": self.factor_dtype.name,
            "displacement_quantum_bohr": self.displacement_quantum_bohr,
            "endpoint_policy": "exclude_i0_and_jR",
            "factor_support_semantics": P23_FACTOR_SUPPORT_SEMANTICS,
            "reverse_edge_policy": "compute_once_then_exact_transpose",
            "image_enumerator": enumerator.snapshot(),
            **query_stats,
            **contraction_stats,
        }
        return node, edge, stats


__all__ = [
    "P23_FACTOR_SUPPORT_SEMANTICS",
    "P23_VNA_ASSEMBLY_SCHEMA",
    "P23_VNA_TABLE_SCHEMA",
    "P23VNAFactorAssembler",
    "P23VNAFactorTableStore",
]
