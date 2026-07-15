"""Output-field specification for multi-expert ensemble stitching.

The :class:`~dptb.nn.build.DistanceEnsembleWrapper` stitches the per-expert
inference outputs of several distance-range experts back into a single result
dictionary.  Historically this was done purely by *guessing* alignment from a
tensor's leading dimension (``src.shape[0] == num_edges``), which silently
dropped every node-aligned output (``node_hamiltonian`` / ``node_overlap`` /
``node_h0`` / ``node_p2`` / ``node_p23`` / ``node_*_blocks`` / onsite blocks)
produced by experts ``1..N``.

This module makes the contract explicit: every relevant key declared in
:mod:`dptb.data._keys` is classified -- by prefix -- into an
:class:`OutputFieldSpec` describing

* ``alignment`` -- whether the leading dimension indexes nodes, edges, a graph
  level quantity, or a scalar; and
* ``merge`` -- how the per-expert values are combined.

The wrapper then iterates the *spec* (not the raw expert output dict) so it can
pick the correct mask (edge- vs node-aligned), validate shapes, and dispatch the
merge deterministically instead of guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

log = logging.getLogger(__name__)

# Allowed vocabularies.  Keep these small and explicit so a typo in a spec is a
# hard error rather than a silent mis-classification.
ALIGNMENTS = ("node", "edge", "graph", "scalar")
MERGES = ("masked_replace", "sum", "mean", "keep_first")

# Alignments whose leading dimension is indexed by a boolean mask.  Graph and
# scalar quantities are not masked (they are shared across experts).
MASKED_ALIGNMENTS = frozenset({"node", "edge"})


class ModelOutputSpecError(Exception):
    """Raised when ensemble output stitching violates the declared spec.

    In *strict* mode this surfaces (a) outputs whose leading dimension does not
    match the declared alignment, and (b) output keys that the spec does not
    declare -- whose alignment would have to be guessed from shape, which is
    outright impossible when ``num_nodes == num_edges``.  Declared fields carry
    a declared alignment, so the node/edge count coincidence never affects
    them.  In permissive mode these become warnings and the offending field is
    skipped (expert-0 value preserved).
    """


@dataclass(frozen=True)
class OutputFieldSpec:
    """Declares how a single model-output field is aligned and merged.

    Parameters
    ----------
    field:
        The output dictionary key (e.g. ``"node_hamiltonian"``).
    alignment:
        One of ``node`` / ``edge`` / ``graph`` / ``scalar``.  ``node`` and
        ``edge`` outputs are index-merged with the corresponding boolean mask;
        ``graph`` and ``scalar`` outputs are shared across experts.
    merge:
        One of ``masked_replace`` / ``sum`` / ``mean`` / ``keep_first``.
    required:
        Purely declarative metadata: whether the field is expected to be present
        in the ensemble output.  Not enforced by the wrapper (kept ``False`` for
        every default field) so it can never break inference on its own.
    """

    field: str
    alignment: str
    merge: str
    required: bool = False

    def __post_init__(self) -> None:
        if self.alignment not in ALIGNMENTS:
            raise ValueError(
                f"OutputFieldSpec('{self.field}'): alignment '{self.alignment}' "
                f"not in {ALIGNMENTS}"
            )
        if self.merge not in MERGES:
            raise ValueError(
                f"OutputFieldSpec('{self.field}'): merge '{self.merge}' "
                f"not in {MERGES}"
            )

    @property
    def is_masked(self) -> bool:
        """True if this field's leading dim is index-merged with a mask."""
        return self.alignment in MASKED_ALIGNMENTS


class ModelOutputSpec:
    """A collection of :class:`OutputFieldSpec` keyed by field name.

    ``strict`` toggles fail-closed behaviour during stitching: undeclared keys,
    shape/alignment mismatches, and node/edge ambiguities raise
    :class:`ModelOutputSpecError` instead of warning and skipping.
    """

    def __init__(self, fields: Dict[str, OutputFieldSpec], strict: bool = False):
        self.fields: Dict[str, OutputFieldSpec] = dict(fields)
        self.strict: bool = bool(strict)

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def __len__(self) -> int:
        return len(self.fields)

    def get(self, key: str) -> Optional[OutputFieldSpec]:
        return self.fields.get(key, None)

    def masked_fields(self) -> Iterable[OutputFieldSpec]:
        """Yield the specs whose alignment is index-merged (node/edge)."""
        for spec in self.fields.values():
            if spec.is_masked:
                yield spec

    def with_strict(self, strict: bool) -> "ModelOutputSpec":
        """Return a shallow copy toggling the strict flag."""
        return ModelOutputSpec(self.fields, strict=strict)


# ---------------------------------------------------------------------------
# Default spec derived from dptb.data._keys
# ---------------------------------------------------------------------------

# Structural / bookkeeping fields that must never be treated as a node- or
# edge-aligned *output* even though some of them (``edge_index``,
# ``edge_type``, ``edge_lengths`` ...) share the ``edge_``/``node_`` prefix.
# These are geometry, indices, PyG collation internals, or MoE routing scalars
# that are shared across experts; the ensemble keeps expert-0's copy.  This set
# replaces the previously hard-coded ``_build_excluded_stitch_keys``.
STRUCTURAL_FIELDS = frozenset(
    {
        # ensemble / MoE bookkeeping
        "expert_idx",
        "expert_edge_mask",
        "expert_node_mask",
        "mean_max_prob",
        "expert_load_cv",
        # PyG collation internals
        "__slices__",
        "__cumsum__",
        "__cat_dims__",
        "__num_nodes_list__",
        "__data_class__",
        "batch",
        "ptr",
        # geometry / indices / structure (identical across experts)
        "pos",
        "positions",
        "cell",
        "pbc",
        "atom_types",
        "atomic_numbers",
        "edge_index",
        "edge_type",
        "edge_vec",
        "edge_lengths",
        "kpoint",
        "kpoints",
    }
)


def _classify_field(field: str) -> OutputFieldSpec:
    """Classify a single field name into an :class:`OutputFieldSpec` by prefix.

    Precedence (first match wins):

    1. structural / bookkeeping keys        -> graph, keep_first
    2. ``*_block_shape`` metadata tensors   -> graph, keep_first
    3. ``node_*``                           -> node,  masked_replace
    4. ``edge_*``                           -> edge,  masked_replace
    5. anything else (graph-level / scalar) -> graph, keep_first

    ``*_block_shape`` is checked *before* the ``node_``/``edge_`` prefixes
    because e.g. ``node_h0_block_shape`` is a tiny graph-constant describing
    block dimensions, not a per-node quantity.
    """
    if field in STRUCTURAL_FIELDS:
        return OutputFieldSpec(field, "graph", "keep_first")
    if field.endswith("_block_shape"):
        return OutputFieldSpec(field, "graph", "keep_first")
    if field.startswith("node_"):
        return OutputFieldSpec(field, "node", "masked_replace")
    if field.startswith("edge_"):
        return OutputFieldSpec(field, "edge", "masked_replace")
    return OutputFieldSpec(field, "graph", "keep_first")


def default_output_spec(strict: bool = False) -> ModelOutputSpec:
    """Build the default :class:`ModelOutputSpec` from :mod:`dptb.data._keys`.

    Every ``*_KEY`` string constant in ``dptb.data._keys`` is enumerated and
    classified by prefix via :func:`_classify_field`.  This covers the
    HAMILTONIAN / OVERLAP / EDGE_ / NODE_ families, the ``*_blocks`` and
    ``*_block_shape`` families, and the structural geometry keys.  A handful of
    non-``_KEY`` structural / bookkeeping names (PyG collation internals, MoE
    routing scalars) are added so they are *declared* (and therefore never
    trigger the strict "undeclared key" path).
    """
    import dptb.data._keys as _keys  # local import: avoid import-order coupling

    fields: Dict[str, OutputFieldSpec] = {}

    # 1) All *_KEY string constants, classified by prefix.
    for attr in dir(_keys):
        if not attr.endswith("_KEY"):
            continue
        value = getattr(_keys, attr)
        if isinstance(value, str):
            fields[value] = _classify_field(value)

    # 2) Non-_KEY structural / bookkeeping fields that show up in batches or the
    #    MoE routing head but are not declared as key constants.
    for name in STRUCTURAL_FIELDS:
        fields.setdefault(name, _classify_field(name))

    return ModelOutputSpec(fields=fields, strict=strict)
