#!/usr/bin/env python3
"""Build nacf-envxc/v1 tables (density projectors, factor tables, pair tables, background layers).

Written by Claude Fable 5.1, 2026-09-20. Offline only. Inputs are per-species AtomicSource npz
files (see dptb.nacf.envxc_tables.AtomicSource.save) and a JSON list of structure species sets;
the tool derives every needed ordered centre|AO pair and unordered A|B pair from those sets.

    python tools/build_nacf_envxc_tables.py --sources SRC_DIR --structures species_sets.json \
        --output ROOT --radial-rank 2 --l-buffer 1 --tail-seeds 1 --density-threshold 1e-7 \
        --distance-step 0.05 --order 128 --workers 64 \
        [--background-nodes nodes.json | --import-background /path/table_manifest_s3.json] \
        [--kinds rhofac envfac envnorm envpair pairmom xcbg] [--resume] [--check-samples 6]

Build identity (fail closed):
  * ``BUILD_IDENTITY.json`` in the output root pins the schema, the numerical settings and the code
    SHA256s of the build that created the root. Every species/table npz embeds that identity plus the
    SHA256 of the atomic source(s) it was built from.
  * A root that already holds artifacts is only touched with ``--resume`` and only when its recorded
    identity equals the requested one; every reused artifact must exist, match the SHA256 recorded in
    ``progress.jsonl`` (or, for species, be self-consistent) and carry the requested identity and
    source SHA256s. Legacy roots without ``BUILD_IDENTITY.json``, changed rank/cutoff/step/order/
    background nodes, changed sources or changed code all stop with an instruction to use a new root.
    Nothing in a mismatched root is rewritten or relabeled.
  * ``--workers 1`` runs in-process (deterministic, no process pool).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dptb.nacf.envxc_tables import (  # noqa: E402
    CENTRE_KINDS, ENVXC_SCHEMA, TABLE_KINDS, AtomicSource, BuildIdentityError, DensityProjectors,
    build_density_projectors, build_identity, build_table_values, check_identity, distance_grid, load_species,
    numerical_settings, read_identity, save_species, save_table, sha256_file, sha256_json, table_filename, table_key,
)

IDENTITY_FILE = "BUILD_IDENTITY.json"
_SOURCES: dict[str, AtomicSource] = {}
_PROJECTORS: dict[str, DensityProjectors] = {}
_ARGS: dict[str, Any] = {}


def _init_worker(source_dir: str, species_dir: str, settings: dict[str, Any]) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    _ARGS.clear()
    _ARGS.update(settings)
    _ARGS["source_dir"] = source_dir
    _ARGS["species_dir"] = species_dir
    _SOURCES.clear()
    _PROJECTORS.clear()


def _source(symbol: str) -> AtomicSource:
    if symbol not in _SOURCES:
        _SOURCES[symbol] = AtomicSource.load(Path(_ARGS["source_dir"]) / f"{symbol}.npz")
    return _SOURCES[symbol]


def _projectors(symbol: str) -> DensityProjectors:
    if symbol not in _PROJECTORS:
        proj = load_species(Path(_ARGS["species_dir"]) / f"{symbol}.npz")
        proj.symbol = symbol
        _PROJECTORS[symbol] = proj
    return _PROJECTORS[symbol]


def _artifact_identity(kind: str, key: str, index, symbols) -> dict[str, Any]:
    return {"schema": ENVXC_SCHEMA, "kind": kind, "key": key, "index": index, "build_identity": _ARGS["build_identity"],
            "sources": {s: _ARGS["source_sha256"][s] for s in symbols}}


def _build_one(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    kind, key, index = job["kind"], job["key"], job.get("index")
    left_symbol, right_symbol = key.split("|")[:2]
    right = _source(right_symbol)
    if kind in CENTRE_KINDS:
        left = _projectors(left_symbol)
        support = left.cutoff_bohr + right.orbital_cutoff_bohr
    else:
        left = _source(left_symbol) if left_symbol != right_symbol else right
        support = left.orbital_cutoff_bohr + right.orbital_cutoff_bohr
    distances = distance_grid(support, float(_ARGS["distance_step"]))
    background = None if index is None else float(_ARGS["background_nodes"][index])
    table = build_table_values(kind, left, right, distances=distances, order=int(_ARGS["order"]), background=background)
    path = Path(_ARGS["output"]) / table_filename(kind, key)
    extra = {} if background is None else {"background_bohr_minus3": background}
    identity = _artifact_identity(kind, key, index, {left_symbol, right_symbol})
    digest = save_table(path, table, identity=identity, **extra)
    return {"kind": kind, "key": key, "index": index, "path": str(path.relative_to(Path(_ARGS["output"]))),
            "sha256": digest, "left_shells": list(table["left_shells"]), "right_shells": list(table["right_shells"]),
            "support_bohr": table["support_bohr"], "knots": int(len(distances)), "bytes": path.stat().st_size,
            "symmetrized": bool(table.get("symmetrized", False)), "seconds": time.perf_counter() - started}


def _check_one(job: dict[str, Any]) -> dict[str, Any]:
    """Interpolation (midpoint) and quadrature-order convergence for one finished table."""
    kind, key, index = job["kind"], job["key"], job.get("index")
    left_symbol, right_symbol = key.split("|")[:2]
    right = _source(right_symbol)
    left = _projectors(left_symbol) if kind in CENTRE_KINDS else (_source(left_symbol) if left_symbol != right_symbol else right)
    path = Path(_ARGS["output"]) / table_filename(kind, key)
    from dptb.data.interfaces.p2_table import RadialBlockTable
    with np.load(path, allow_pickle=False) as z:
        table = RadialBlockTable(z["distances"], z["values"], tuple(int(x) for x in z["left_shells"]),
                                 tuple(int(x) for x in z["right_shells"]), float(z["support_bohr"]))
    background = None if index is None else float(_ARGS["background_nodes"][index])
    rng = np.random.default_rng(abs(hash((kind, key, index))) % (2**32))
    checks = []
    support = table.support_bohr
    for d in rng.uniform(0.3, max(0.4, support - 0.3), 3):
        d = float(d)
        grid_lo = build_table_values(kind, left, right, distances=np.array([0.0, d, support]), order=int(_ARGS["order"]), background=background)["values"][1]
        grid_hi = build_table_values(kind, left, right, distances=np.array([0.0, d, support]), order=int(_ARGS["order"]) + 64, background=background)["values"][1]
        pred = table._interpolate(d)
        checks.append({"d_bohr": d, "interpolation_max": float(np.max(np.abs(pred - grid_hi))),
                       "quadrature_max": float(np.max(np.abs(grid_hi - grid_lo))), "scale_max": float(np.max(np.abs(grid_hi)))})
    return {"kind": kind, "key": key, "index": index, "checks": checks}


def _load_structures(path: Path) -> list[list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("structures") or data.get("species_sets") or data.get("rows")
    sets = []
    for row in data:
        symbols = row["symbols"] if isinstance(row, dict) else row
        sets.append(sorted(set(str(s) for s in symbols)))
    return sets


def _jobs(structures: list[list[str]], kinds: list[str], n_layers: int) -> list[dict[str, Any]]:
    centre_pairs, unordered = set(), set()
    for symbols in structures:
        for a in symbols:
            for b in symbols:
                centre_pairs.add((a, b))
                unordered.add(tuple(sorted((a, b))))
    jobs = []
    for kind in kinds:
        if kind in CENTRE_KINDS:
            for k, a in sorted(centre_pairs):
                jobs.append({"kind": kind, "key": table_key(kind, k, a)})
        elif kind == "xcbg":
            for a, b in sorted(unordered):
                for index in range(n_layers):
                    jobs.append({"kind": kind, "key": table_key(kind, a, b, index), "index": index})
        else:
            for a, b in sorted(unordered):
                jobs.append({"kind": kind, "key": table_key(kind, a, b)})
    return jobs


def _import_background(manifest_path: Path) -> tuple[list[float], dict[str, dict[str, Any]], str]:
    """Reference round-4 background layers (nacf-background-pair-table/s3-v1) without copying them."""
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not m.get("complete"):
        raise ValueError("imported background manifest is not complete")
    nodes = [float(x) for x in m["nodes"]]
    rows: dict[str, dict[str, Any]] = {}
    for row in m["rows"]:
        a, b = row["pair"].split("|")
        key = table_key("xcbg", a, b, int(row["index"]))
        path = Path(row["file"])
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        rows[key] = {"path": str(path), "sha256": row.get("sha256") or sha256_file(path), "imported": True,
                     "left_shells": row.get("left_shells"), "right_shells": row.get("right_shells"),
                     "support_bohr": row.get("support_bohr"), "source_manifest": str(manifest_path)}
    return nodes, rows, sha256_file(manifest_path)


def _root_has_artifacts(output: Path) -> bool:
    if (output / "manifest.json").exists() or (output / "progress.jsonl").exists():
        return True
    return any(output.rglob("*.npz"))


def _identity_diff(recorded: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    diff = {}
    for section in ("settings", "code_identity"):
        old, new = recorded.get(section, {}), current.get(section, {})
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                diff[f"{section}.{key}"] = {"recorded": old.get(key), "requested": new.get(key)}
    return diff


def run(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", type=Path, required=True, help="directory of AtomicSource npz files (<symbol>.npz)")
    p.add_argument("--structures", type=Path, required=True, help="JSON: list of species lists (or {'structures': [...]})")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kinds", nargs="+", default=list(TABLE_KINDS), choices=list(TABLE_KINDS))
    p.add_argument("--radial-rank", type=int, default=2, help="radial projectors per l (cheap default 2; 3 is the richer option)")
    p.add_argument("--l-buffer", type=int, default=1, help="angular momenta beyond the PAO l_max (cheap default 1; 2 is the richer option)")
    p.add_argument("--tail-seeds", type=int, default=1)
    p.add_argument("--density-threshold", type=float, default=1e-7)
    p.add_argument("--projector-cutoff", type=float, default=None, help="override the density cutoff for q (bohr); an explicit extra truncation")
    p.add_argument("--distance-step", type=float, default=0.05)
    p.add_argument("--order", type=int, default=128)
    p.add_argument("--background-nodes", type=Path, default=None, help="JSON list of background densities (first must be 0)")
    p.add_argument("--import-background", type=Path, default=None, help="round-4 table_manifest_s3.json to reference instead of rebuilding xcbg")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-samples", type=int, default=0, help="number of finished tables to spot-check (interpolation/quadrature)")
    a = p.parse_args(argv)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    started = time.time()
    output = a.output.resolve()
    structures = _load_structures(a.structures)
    symbols = sorted({s for row in structures for s in row})
    missing = [s for s in symbols if not (a.sources / f"{s}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"missing atomic sources: {missing}")
    # ---- background nodes ------------------------------------------------------------------------
    nodes: list[float] | None = None
    imported: dict[str, dict[str, Any]] = {}
    import_sha = None
    kinds = list(a.kinds)
    if a.import_background is not None:
        nodes, imported, import_sha = _import_background(a.import_background)
        kinds = [k for k in kinds if k != "xcbg"]
    elif a.background_nodes is not None:
        nodes = [float(x) for x in json.loads(a.background_nodes.read_text())]
        if nodes[0] != 0.0 or np.any(np.diff(nodes) <= 0):
            raise ValueError("background nodes must start with 0 and increase")
    elif "xcbg" in kinds:
        raise ValueError("xcbg requested without --background-nodes or --import-background")
    # ---- immutable build identity ------------------------------------------------------------------
    code_identity = {"envxc_tables.py": sha256_file(ROOT_DIR / "dptb/nacf/envxc_tables.py"),
                     "build_nacf_envxc_tables.py": sha256_file(__file__)}
    settings = {"radial_rank": a.radial_rank, "l_buffer": a.l_buffer, "tail_seeds": a.tail_seeds, "density_threshold": a.density_threshold,
                "projector_cutoff": a.projector_cutoff, "grid_step": 0.005, "distance_step": a.distance_step, "order": a.order,
                "background_nodes": nodes, "import_background_sha256": import_sha}
    build_id = build_identity(settings, code_identity)
    identity_record = {"schema": ENVXC_SCHEMA, "build_identity": build_id, "settings": numerical_settings(settings),
                       "code_identity": code_identity, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    source_sha = {s: sha256_file(a.sources / f"{s}.npz") for s in symbols}
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / IDENTITY_FILE
    if _root_has_artifacts(output):
        if not identity_path.is_file():
            raise BuildIdentityError(f"{output} holds artifacts but no {IDENTITY_FILE}: legacy/unidentified root. "
                                     "Nothing was changed; choose a new --output root.")
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded.get("build_identity") != build_id:
            raise BuildIdentityError(f"{output} was built with a different identity {recorded.get('build_identity')!r} "
                                     f"(requested {build_id!r}); differences: {json.dumps(_identity_diff(recorded, identity_record), sort_keys=True)}. "
                                     "Nothing was changed; choose a new --output root.")
        if not a.resume:
            raise BuildIdentityError(f"{output} already holds artifacts of this identity; pass --resume to continue it or choose a new --output root.")
    elif identity_path.is_file():
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded.get("build_identity") != build_id:
            raise BuildIdentityError(f"{output} carries identity {recorded.get('build_identity')!r} without artifacts; choose a new --output root.")
    else:
        identity_path.write_text(json.dumps(identity_record, indent=1, sort_keys=True), encoding="utf-8")
    species_dir = output / "species"
    species_dir.mkdir(exist_ok=True)
    progress_path = output / "progress.jsonl"
    recorded_rows: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                recorded_rows[f"{row['kind']}:{row['key']}"] = row
    # ---- species projectors (reuse only with verified identity) ------------------------------------
    species_rows: dict[str, dict[str, Any]] = {}
    sources_identity: dict[str, Any] = {}
    species_progress = []
    for s in symbols:
        source = AtomicSource.load(a.sources / f"{s}.npz")
        sources_identity[s] = {"source_sha256": source_sha[s], **source.identity}
        path = species_dir / f"{s}.npz"
        identity = {"schema": ENVXC_SCHEMA, "kind": "species", "key": s, "build_identity": build_id, "sources": {s: source_sha[s]}}
        if path.is_file():
            if not a.resume:
                raise BuildIdentityError(f"{path} exists; pass --resume or choose a new --output root")
            label = f"species:{s}"
            proj = load_species(path)
            check_identity(proj.identity, build_id=build_id, sources={s: source_sha[s]}, label=label)
            recorded = recorded_rows.get(f"species:{s}")
            if recorded is not None and sha256_file(path) != recorded["sha256"]:
                raise BuildIdentityError(f"{label}: file SHA256 differs from the recorded build (corrupted or replaced); choose a new --output root")
            proj.symbol = s
        else:
            proj = build_density_projectors(source, radial_rank=a.radial_rank, l_buffer=a.l_buffer, tail_seeds=a.tail_seeds,
                                            density_threshold=a.density_threshold, cutoff_bohr=a.projector_cutoff)
            proj.metadata["symbol"] = s
            proj.identity = identity
            save_species(path, proj, identity=identity)
            species_progress.append({"kind": "species", "key": s, "path": str(path.relative_to(output)), "sha256": sha256_file(path)})
        species_rows[s] = {"array_path": str(path.relative_to(output)), "array_sha256": sha256_file(path),
                           "q_shells": [int(l) for l in proj.q_l], "q_norb": int(proj.norb), "q_cutoff_bohr": float(proj.cutoff_bohr),
                           "orbital_shells": list(source.shells), "orbital_norb": int(source.norb),
                           "orbital_cutoff_bohr": float(source.orbital_cutoff_bohr), "z_valence": float(source.z_valence),
                           "metadata": proj.metadata, "identity": proj.identity}
        print("SPECIES", s, "q_norb", proj.norb, "cutoff", round(proj.cutoff_bohr, 3), "beyond_cutoff_e", round(proj.metadata["electrons_beyond_cutoff"], 6), flush=True)
    worker_settings = dict(settings, output=str(output), build_identity=build_id, source_sha256=source_sha)
    # ---- jobs --------------------------------------------------------------------------------------
    jobs = _jobs(structures, kinds, 0 if nodes is None else len(nodes))
    done: dict[str, dict[str, Any]] = {}
    for job in jobs:
        ident = f"{job['kind']}:{job['key']}"
        row = recorded_rows.get(ident)
        if row is None:
            continue
        path = output / row["path"]
        label = f"{ident} ({row['path']})"
        if not path.is_file():
            raise BuildIdentityError(f"{label}: recorded in progress.jsonl but missing on disk; choose a new --output root")
        if sha256_file(path) != row["sha256"]:
            raise BuildIdentityError(f"{label}: file SHA256 differs from the recorded build (corrupted or replaced); choose a new --output root")
        left_symbol, right_symbol = job["key"].split("|")[:2]
        check_identity(read_identity(path), build_id=build_id, sources={x: source_sha[x] for x in {left_symbol, right_symbol}}, label=label)
        done[ident] = row
    pending = [j for j in jobs if f"{j['kind']}:{j['key']}" not in done]
    print("JOBS total", len(jobs), "reused", len(done), "pending", len(pending), "workers", a.workers, flush=True)
    failures = []
    with progress_path.open("a", encoding="utf-8") as log:
        for row in species_progress:
            log.write(json.dumps(row) + "\n")
        log.flush()
        if a.workers <= 1:
            _init_worker(str(a.sources), str(species_dir), worker_settings)
            for n, job in enumerate(pending, 1):
                try:
                    row = _build_one(job)
                except Exception as exc:
                    failures.append({"job": job, "error": repr(exc)})
                    print("FAILED", job, repr(exc), flush=True)
                    continue
                done[f"{row['kind']}:{row['key']}"] = row
                log.write(json.dumps(row) + "\n"); log.flush()
                if n % 25 == 0 or n == len(pending):
                    print("PROGRESS", n, "/", len(pending), row["kind"], row["key"], round(row["seconds"], 2), "s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker, initargs=(str(a.sources), str(species_dir), worker_settings)) as pool:
                futures = {pool.submit(_build_one, j): j for j in pending}
                for n, fut in enumerate(as_completed(futures), 1):
                    job = futures[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:  # record and continue; manifest stays incomplete
                        failures.append({"job": job, "error": repr(exc)})
                        print("FAILED", job, repr(exc), flush=True)
                        continue
                    done[f"{row['kind']}:{row['key']}"] = row
                    log.write(json.dumps(row) + "\n"); log.flush()
                    if n % 25 == 0 or n == len(pending):
                        print("PROGRESS", n, "/", len(pending), row["kind"], row["key"], round(row["seconds"], 2), "s", flush=True)
    # ---- checks ------------------------------------------------------------------------------------
    checks = []
    if a.check_samples > 0 and done:
        sample = [done[k] for k in sorted(done)[:: max(1, len(done) // a.check_samples)]][: a.check_samples]
        if a.workers <= 1:
            _init_worker(str(a.sources), str(species_dir), worker_settings)
            checks = [_check_one(j) for j in sample]
        else:
            with ProcessPoolExecutor(max_workers=min(a.workers, len(sample)), initializer=_init_worker, initargs=(str(a.sources), str(species_dir), worker_settings)) as pool:
                for fut in as_completed([pool.submit(_check_one, j) for j in sample]):
                    checks.append(fut.result())
    # ---- manifest ----------------------------------------------------------------------------------
    tables: dict[str, dict[str, Any]] = {k: {} for k in kinds}
    for row in done.values():
        tables.setdefault(row["kind"], {})[row["key"]] = {k: row[k] for k in ("path", "sha256", "left_shells", "right_shells", "support_bohr", "knots", "bytes") if k in row}
    if imported:
        tables["xcbg"] = imported
    complete = not failures and all(f"{j['kind']}:{j['key']}" in done for j in jobs)
    manifest = {
        "schema": ENVXC_SCHEMA, "complete": complete, "length_unit": "bohr", "density_unit": "bohr^-3", "xc_energy_unit": "eV",
        "harmonic_convention": "deeptb_abacus_real", "endpoint_policy": "exclude_i0_and_jR", "interpolation": "cubic",
        "xc_functional": "LDA-PZ81 unpolarized; density = normalized neutral valence + unscaled NLCC",
        "density_operator": "rho_K ~= sum_h |rho_K p_h> eps_h <p_h rho_K|, p_h Gram-Schmidt in the rho_K metric",
        "build_identity": build_id, "identity_file": IDENTITY_FILE, "settings": settings, "numerical_settings": numerical_settings(settings),
        "background_nodes": nodes, "species": species_rows, "tables": tables, "structures": structures, "sources": sources_identity,
        "code_identity": code_identity, "code_identity_sha256": sha256_json(code_identity), "failures": failures, "checks": checks,
        "table_count": sum(len(v) for v in tables.values()), "total_bytes": int(sum(r.get("bytes", 0) for r in done.values())),
        "imported_background_bytes": int(sum(Path(r["path"]).stat().st_size for r in imported.values())) if imported else 0,
        "build_wall_seconds": time.time() - started, "build_worker_seconds": float(sum(r["seconds"] for r in done.values() if "seconds" in r)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    print("MANIFEST", "complete" if complete else "INCOMPLETE", "tables", manifest["table_count"], "bytes", manifest["total_bytes"],
          "wall_s", round(manifest["build_wall_seconds"], 1), "identity", build_id[:16], flush=True)
    return 0 if complete else 1


def main(argv=None) -> int:
    try:
        return run(argv)
    except BuildIdentityError as exc:
        print("FAIL_CLOSED", exc, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
