"""Endpoint metrics: the optimized objective and the public onsite/hopping/loss
triplet stay correctly separated and correctly averaged.

Covers per-iteration/epoch scheduler metric selection (Trainer prefers the
optimized objective, MultiTrainer the endpoint loss, flow always prefers the
endpoint), the fail-closed ``validation()`` return, and per-key valid-batch
counts so a throttled feature-compatible metric is not diluted by batches
that never fired it.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from dptb.nnops import trainer as trainer_mod
from dptb.nnops.base_trainer import BaseTrainer
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.nnops.trainer import Trainer
from dptb.tests._trainer_probes import (
    DistinctEndpointLoss,
    FakeBatch,
    RecordingMetricScheduler,
    ScalarLoss,
    make_fake_trainer,
    make_validation_trainer,
)

BASIS = {"H": "1s", "O": "1s1p"}


def _data() -> dict:
    """Fresh H-O block payload: zero predictions against nonzero target blocks."""
    max_norb = 4  # 1s1p union
    pred_node = torch.zeros(2, max_norb, max_norb)
    target_node = torch.zeros(2, max_norb, max_norb)
    target_node[0, 0, 0] = 0.5  # H onsite 1x1
    target_node[1, :4, :4] = 0.25  # O onsite 4x4
    pred_edge = torch.zeros(2, max_norb, max_norb)
    target_edge = torch.zeros(2, max_norb, max_norb)
    target_edge[0, :1, :4] = 1.0  # H->O 1x4
    target_edge[1, :4, :1] = -1.0  # O->H 4x1
    return {
        "node_hamil_blocks": pred_node, "edge_hamil_blocks": pred_edge,
        "atom_types": torch.tensor([[0], [1]]), "atomic_numbers": torch.tensor([1, 8]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "node_delta_hamil_blocks": target_node, "edge_delta_hamil_blocks": target_edge,
        "node_delta_hamil_block_shape": torch.tensor([[1, 1], [4, 4]]),
        "edge_delta_hamil_block_shape": torch.tensor([[1, 4], [4, 1]]),
    }


def _criterion(**kwargs):
    from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss

    opts = dict(basis=BASIS, optimization="block_mae", block_reduction="global")
    opts.update(kwargs)
    return HamilBlockwiseNexTHamLoss(**opts)


# ---------------------------------------------------------------------------
# scalar vs endpoint-triplet criteria stay correctly separated
# ---------------------------------------------------------------------------
def test_nonflow_scalar_criterion_survives_train_and_validation_without_triplet(monkeypatch):
    """A non-Hamiltonian criterion has no onsite/hopping side effects; neither the
    train iteration nor validation must require or fabricate them."""
    trainer, observed_states = make_fake_trainer(monkeypatch)
    train_loss = ScalarLoss()
    trainer.train_lossfunc = train_loss

    objective = trainer.iteration(FakeBatch("train", 2.0))

    assert objective.item() == pytest.approx(2.0)
    assert train_loss.calls == ["train"]
    train_state = observed_states[0][2]
    assert train_state["train_loss"].item() == pytest.approx(2.0)
    assert train_state["train_loss_opt"].item() == pytest.approx(2.0)
    assert "train_onsite_loss" not in train_state
    assert "train_hopping_loss" not in train_state

    validation_loss = ScalarLoss()
    validation_trainer = make_validation_trainer(monkeypatch, validation_loss)
    public_validation_loss = validation_trainer.validation(fast=True)

    assert public_validation_loss.item() == pytest.approx(3.0)
    assert validation_loss.calls == ["validation"]
    validation_state = validation_trainer._last_flow_validation_state
    assert validation_state["validation_loss"].item() == pytest.approx(3.0)
    assert "validation_onsite_loss" not in validation_state
    assert "validation_hopping_loss" not in validation_state


def test_nonflow_validation_keeps_endpoint_public_and_objective_separate(monkeypatch):
    trainer = make_validation_trainer(monkeypatch, DistinctEndpointLoss(endpoint_loss=23.0))

    public_validation_loss = trainer.validation(fast=True)

    assert public_validation_loss.item() == pytest.approx(23.0)
    state = trainer._last_flow_validation_state
    assert state["validation_loss"].item() == pytest.approx(23.0)
    assert state["validation_loss_opt"].item() == pytest.approx(3.0)
    assert state["validation_onsite_loss"].item() == pytest.approx(24.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# per-iteration scheduler: the objective drives LR, the endpoint stays public
# ---------------------------------------------------------------------------
def test_nonflow_per_iteration_scheduler_uses_objective_while_public_loss_is_endpoint(monkeypatch):
    trainer, observed_states = make_fake_trainer(monkeypatch)
    trainer.train_lossfunc = DistinctEndpointLoss(endpoint_loss=13.0)
    trainer.update_lr_per_iter = True
    trainer.iter = 2
    trainer.stats = {
        "train_loss": {"latest_avg_iter_loss": torch.tensor(91.0)},
        "train_loss_opt": {"latest_avg_iter_loss": torch.tensor(17.0)},
    }
    trainer.lr_scheduler = RecordingMetricScheduler()

    objective = trainer.iteration(FakeBatch("train", 2.0))

    assert trainer.lr_scheduler.metrics == pytest.approx([17.0])
    assert objective.item() == pytest.approx(2.0)
    state = observed_states[0][2]
    assert state["train_loss"].item() == pytest.approx(13.0)
    assert state["train_loss_opt"].item() == pytest.approx(2.0)


def test_flow_per_iteration_scheduler_keeps_endpoint_metric(monkeypatch):
    trainer, observed_states = make_fake_trainer(monkeypatch)
    trainer.flow_cfm = SimpleNamespace(enabled=True)
    trainer.update_lr_per_iter = True
    trainer.iter = 2
    trainer.stats = {
        "train_loss": {"latest_avg_iter_loss": torch.tensor(91.0)},
        "train_loss_opt": {"latest_avg_iter_loss": torch.tensor(17.0)},
    }
    trainer.lr_scheduler = RecordingMetricScheduler()

    def fake_flow_loss(batch, lossfunc, *, use_flow=True, allow_self_consistency=True):
        trainer._last_flow_state = {
            "train_loss": torch.tensor(13.0),
            "train_onsite_loss": torch.tensor(14.0),
            "train_hopping_loss": torch.tensor(15.0),
        }
        return trainer.model.weight * batch.x

    trainer._loss_on_batch = fake_flow_loss

    objective = trainer.iteration(FakeBatch("train", 2.0))

    assert trainer.lr_scheduler.metrics == pytest.approx([91.0])
    assert objective.item() == pytest.approx(2.0)
    state = observed_states[0][2]
    assert state["train_loss"].item() == pytest.approx(13.0)
    assert state["train_loss_opt"].item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# epoch scheduler: Trainer prefers the objective, MultiTrainer the endpoint,
# flow always prefers the endpoint regardless of the class default
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("trainer_cls", "flow_enabled", "expected_metric"),
    [(Trainer, False, 19.0), (Trainer, True, 11.0), (MultiTrainer, False, 11.0)],
    ids=["single_trainer_prefers_objective", "single_trainer_flow_prefers_endpoint",
         "multi_trainer_prefers_endpoint"],
)
def test_epoch_scheduler_respects_single_trainer_objective_contract(trainer_cls, flow_enabled, expected_metric):
    scheduler = RecordingMetricScheduler()
    trainer = SimpleNamespace(
        ep=1, plugin_queues={}, epoch=lambda: None, call_plugins=lambda **kwargs: None, update=lambda: None,
        update_lr_per_iter=False,
        scheduler_metric_prefers_objective=trainer_cls.scheduler_metric_prefers_objective,
        lr_scheduler=scheduler,
        flow_cfm=SimpleNamespace(enabled=flow_enabled),
        stats={
            "validation_loss": {"epoch_mean": torch.tensor(11.0)},
            "validation_loss_opt": {"epoch_mean": torch.tensor(19.0)},
            "train_loss": {"epoch_mean": torch.tensor(31.0)},
            "train_loss_opt": {"epoch_mean": torch.tensor(37.0)},
        },
    )

    BaseTrainer.run(trainer, epochs=1)

    assert scheduler.metrics == pytest.approx([expected_metric])


# ---------------------------------------------------------------------------
# validation()'s fail-closed return: prefer legacy validation_loss, else the
# smallest-n euler-compatible loss, else the accumulated loss unchanged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("state", "expected_value", "same_object_as"),
    [
        ({"validation_compatible_euler_3_loss": 7.0}, 7.0, None),
        ({"validation_loss": 5.0, "validation_compatible_euler_1_loss": 2.0,
          "validation_compatible_euler_3_loss": 7.0}, 5.0, "validation_loss"),
        ({"validation_flow_random_t_loss": 9.0}, None, "accumulated"),
        ({}, None, "accumulated"),
        ({"validation_compatible_euler_5_loss": 50.0, "validation_compatible_euler_2_loss": 20.0,
          "validation_compatible_euler_10_loss": 100.0}, 20.0, None),
        ({"validation_compatible_euler_x_loss": 99.0, "validation_compatible_euler_4_loss": 4.0}, 4.0, None),
    ],
    ids=["euler_used_when_legacy_absent", "legacy_takes_precedence_byte_identical",
         "falls_through_no_compatible_key", "falls_through_empty_state",
         "smallest_num_steps_wins_numeric_not_lexical", "non_numeric_suffix_ignored"],
)
def test_validation_return_resolves_fail_closed(state, expected_value, same_object_as):
    tensors = {key: torch.tensor(value) for key, value in state.items()}
    trainer = Trainer.__new__(Trainer)
    trainer._last_flow_validation_state = tensors
    accumulated = torch.tensor(3.3)

    result = trainer._resolve_validation_return(accumulated)

    assert result.item() == pytest.approx(expected_value if expected_value is not None else 3.3)
    if same_object_as == "accumulated":
        assert result is accumulated
    elif same_object_as is not None:
        assert result is tensors[same_object_as]


def test_validation_return_falls_through_when_attribute_entirely_missing():
    accumulated = torch.tensor(1.25)
    bare = Trainer.__new__(Trainer)
    assert bare._resolve_validation_return(accumulated) is accumulated


# ---------------------------------------------------------------------------
# per-key valid-batch counts: a throttled feature-compatible metric drops out
# of both the numerator and the denominator instead of diluting the average
# ---------------------------------------------------------------------------
class _StubCriterion:
    """Bare loss-module stub exposing only the feature-compatible side effects.

    A firing step exposes last_onsite_loss/last_hopping_loss as tensors; a
    throttled step leaves them None. It has no last_onsite_l1_sum/count, so
    the only onsite/hopping signal flows through the weighted-sum path here.
    """

    def __init__(self, onsite, hopping):
        self.last_onsite_loss = onsite
        self.last_hopping_loss = hopping
        self.last_z_loss = None
        self.expert_load_cv = None


def _minimal_multitrainer() -> MultiTrainer:
    mt = MultiTrainer.__new__(MultiTrainer)
    mt.dtype = torch.float64
    mt.device = torch.device("cpu")
    return mt


_ACTIVE_NODES = 4.0
_ACTIVE_EDGES = 6.0
_TRUE_METRIC = 2.0


def _payload(mt: MultiTrainer, onsite_val, hopping_val):
    """Drive the real _snapshot_loss_metrics + _build_train_payload seam.

    Only the model-dependent _run_one_expert_loss is stubbed; the snapshot and
    the weighted-sum/weight aggregation under test run for real.
    """
    stub = _StubCriterion(
        None if onsite_val is None else torch.tensor(onsite_val, dtype=torch.float64),
        None if hopping_val is None else torch.tensor(hopping_val, dtype=torch.float64),
    )

    def _fake_run_one_expert_loss(**kwargs):
        out = {
            "loss": torch.zeros((), dtype=torch.float64),
            "active_nodes": torch.tensor(_ACTIVE_NODES, dtype=torch.float64),
            "active_edges": torch.tensor(_ACTIVE_EDGES, dtype=torch.float64),
        }
        out.update(mt._snapshot_loss_metrics(stub))
        return out

    mt._run_one_expert_loss = _fake_run_one_expert_loss
    mt.train_lossfunc = stub
    return mt._build_train_payload(batch_dict=None, batch_info=None, expert_idx=0, range_dis=None)


def _aggregate_onsite_hopping(mt: MultiTrainer, payloads):
    pack = torch.zeros(MultiTrainer._PACK_LEN, dtype=mt.dtype, device=mt.device)
    for p in payloads:
        pack = pack + mt._make_step_pack(p)
    onsite = (pack[MultiTrainer._P_ONSITE_WEIGHTED_SUM] / pack[MultiTrainer._P_ACTIVE_NODES_SUM].clamp_min(1.0)).item()
    hopping = (pack[MultiTrainer._P_HOPPING_WEIGHTED_SUM] / pack[MultiTrainer._P_ACTIVE_EDGES_SUM].clamp_min(1.0)).item()
    return onsite, hopping, pack


def test_snapshot_marks_throttled_metrics_none_with_real_criterion():
    mt = _minimal_multitrainer()
    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=2)

    crit(_data())  # call 1 fires
    fired = mt._snapshot_loss_metrics(crit)
    assert fired["onsite"] is not None and torch.isfinite(fired["onsite"])
    assert fired["hopping"] is not None and torch.isfinite(fired["hopping"])
    assert torch.equal(fired["onsite"], crit.last_onsite_loss)
    assert torch.equal(fired["hopping"], crit.last_hopping_loss)

    crit(_data())  # call 2 throttled -> attrs None -> snapshot invalid
    skipped = mt._snapshot_loss_metrics(crit)
    assert skipped["onsite"] is None
    assert skipped["hopping"] is None


def test_aggregation_not_diluted_by_throttled_batch():
    mt = _minimal_multitrainer()

    # interval=1 analog: both batches fire.
    onsite1, hopping1, _ = _aggregate_onsite_hopping(
        mt, [_payload(mt, _TRUE_METRIC, _TRUE_METRIC), _payload(mt, _TRUE_METRIC, _TRUE_METRIC)]
    )
    assert onsite1 == pytest.approx(_TRUE_METRIC)
    assert hopping1 == pytest.approx(_TRUE_METRIC)

    # interval=2 analog: batch 1 fires, batch 2 throttled.
    p_fire = _payload(mt, _TRUE_METRIC, _TRUE_METRIC)
    p_skip = _payload(mt, None, None)
    onsite2, hopping2, pack2 = _aggregate_onsite_hopping(mt, [p_fire, p_skip])

    # Not diluted toward 1.0 -- identical to interval=1; only the update cadence differs.
    assert onsite2 == pytest.approx(_TRUE_METRIC)
    assert hopping2 == pytest.approx(_TRUE_METRIC)

    # The throttled batch contributes ZERO numerator and ZERO weight (count) ...
    assert p_skip["onsite_weighted_sum"].item() == 0.0
    assert p_skip["hopping_weighted_sum"].item() == 0.0
    assert p_skip["onsite_weight"].item() == 0.0
    assert p_skip["hopping_weight"].item() == 0.0
    # ... while its raw active_nodes/active_edges telemetry is untouched.
    assert p_skip["active_nodes"].item() == pytest.approx(_ACTIVE_NODES)
    assert p_skip["active_edges"].item() == pytest.approx(_ACTIVE_EDGES)
    # A firing batch's gated weight equals the raw active count.
    assert p_fire["onsite_weight"].item() == pytest.approx(_ACTIVE_NODES)
    assert p_fire["hopping_weight"].item() == pytest.approx(_ACTIVE_EDGES)
    # The aggregated pack denominators hold only the firing batch's count.
    assert pack2[MultiTrainer._P_ACTIVE_NODES_SUM].item() == pytest.approx(_ACTIVE_NODES)
    assert pack2[MultiTrainer._P_ACTIVE_EDGES_SUM].item() == pytest.approx(_ACTIVE_EDGES)
    assert pack2[MultiTrainer._P_ONSITE_WEIGHTED_SUM].item() == pytest.approx(_TRUE_METRIC * _ACTIVE_NODES)


def test_compute_compatible_state_from_pack_reports_true_metric_when_throttled():
    mt = _minimal_multitrainer()
    pack = mt._make_step_pack(_payload(mt, _TRUE_METRIC, _TRUE_METRIC)) + mt._make_step_pack(
        _payload(mt, None, None)
    )

    state = mt._compute_compatible_state_from_pack(pack, criterion=_StubCriterion(None, None), prefix="train")

    assert state is not None
    assert state["train_onsite_loss"].item() == pytest.approx(_TRUE_METRIC)
    assert state["train_hopping_loss"].item() == pytest.approx(_TRUE_METRIC)


class _ValidationBatch:
    """The batch surface Trainer.validation() reads before AtomicData conversion."""

    __slices__ = {}
    __cumsum__ = {}
    __cat_dims__ = {}
    __num_nodes_list__ = []
    __data_class__ = None

    def __init__(self):
        self.payload = _data()

    def to(self, device):
        return self


class _IdentityModel:
    """A no-op model: passes the block payload through unchanged."""

    def eval(self):
        pass

    def __call__(self, batch):
        return dict(batch)


def test_h10_trainer_validation_per_key_count_not_diluted(monkeypatch):
    """Trainer.validation over 2 batches with the real criterion throttled to
    interval=2: onsite/hopping fire on only 1 of 2 batches and must be divided
    by that 1 (undiluted), not by num_batches=2."""
    reference = _criterion(log_feature_compatible=True)
    reference(_data())
    expected_onsite = reference.last_onsite_loss.item()
    expected_hopping = reference.last_hopping_loss.item()

    monkeypatch.setattr(trainer_mod.AtomicData, "to_AtomicDataDict", lambda batch: dict(batch.payload))
    trainer = Trainer.__new__(Trainer)
    trainer.dtype = torch.float64
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = SimpleNamespace(enabled=False)
    trainer.model = _IdentityModel()
    trainer.validation_lossfunc = _criterion(log_feature_compatible=True, log_feature_compatible_interval=2)
    trainer.validation_loader = [_ValidationBatch(), _ValidationBatch()]

    trainer.validation(fast=False)

    state = trainer._last_flow_validation_state
    assert state["validation_onsite_loss"].item() == pytest.approx(expected_onsite)
    assert state["validation_hopping_loss"].item() == pytest.approx(expected_hopping)
    # not diluted by the uniform num_batches=2 divisor
    assert state["validation_onsite_loss"].item() != pytest.approx(expected_onsite / 2.0)


class _NullTagger:
    def tag(self, *args, **kwargs):
        return contextlib.nullcontext()


def test_h10b_multitrainer_validation_per_key_count_not_diluted(monkeypatch):
    """MultiTrainer.validation twin of h10: onsite present on 1 of 2 batches must
    report 2.0 (divided by its OWN contributing-batch count), not 1.0 (value /
    num_batches). Exercises the real validation() accumulation + divisor."""
    mt = _minimal_multitrainer()
    mt.model = SimpleNamespace(eval=lambda: None)
    mt.validation_loader = [object(), object()]  # two batches
    mt.validation_loader_generator = None
    mt.distributed_expert = False
    mt.flow_cfm = None
    mt.endpoint_loss_mode = "full_forward"
    mt.iter = 0
    mt._tagger = _NullTagger()
    mt.validation_lossfunc = _StubCriterion(None, None)

    monkeypatch.setattr(mt, "_prepare_batch_bundle", lambda batch, with_lengths=True: (None, None))
    monkeypatch.setattr(mt, "_run_full_batch_loss", lambda *a, **k: torch.zeros((), dtype=mt.dtype))
    snapshots = iter([
        {"onsite": torch.tensor(2.0, dtype=mt.dtype), "hopping": torch.tensor(2.0, dtype=mt.dtype)},
        {"onsite": None, "hopping": None},  # throttled/omitted this batch
    ])
    monkeypatch.setattr(mt, "_snapshot_loss_metrics", lambda crit: next(snapshots))

    mt.validation(fast=False)

    state = mt._last_flow_validation_state
    # Un-diluted: divided by its own count (1 firing batch), not num_batches (2).
    assert state["validation_onsite_loss"].item() == pytest.approx(2.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(2.0)


def _stub(value):
    return _StubCriterion(
        None if value is None else torch.tensor(value, dtype=torch.float64),
        None if value is None else torch.tensor(value, dtype=torch.float64),
    )


def _expert_run_output(mt, stub, active_nodes, active_edges):
    out = {
        "loss": torch.zeros((), dtype=torch.float64),
        "active_nodes": torch.tensor(active_nodes, dtype=torch.float64),
        "active_edges": torch.tensor(active_edges, dtype=torch.float64),
    }
    out.update(mt._snapshot_loss_metrics(stub))
    return out


def test_h11_expert_display_metric_uses_gated_denominator_and_main_telemetry():
    """A throttled reference-loss metric affects the backward objective but stays
    isolated from the main-batch endpoint metrics and raw activity telemetry."""
    mt = _minimal_multitrainer()
    mt.distributed_expert = False

    outputs = iter([_expert_run_output(mt, _stub(2.0), 4.0, 6.0),
                    _expert_run_output(mt, _stub(None), 4.0, 6.0)])
    mt._run_one_expert_loss = lambda **kwargs: next(outputs)
    mt.train_lossfunc = _stub(2.0)
    payload = mt._build_train_payload(
        batch_dict=None, batch_info=None, expert_idx=0, range_dis=None,
        ref_batch_dict={"stub": True}, ref_batch_info=None,
    )

    assert payload["expert_onsite"].item() == pytest.approx(2.0)
    assert payload["expert_hopping"].item() == pytest.approx(2.0)
    # gated denominators drop the throttled ref (weight 0), not raw 4+4.
    assert payload["onsite_weight"].item() == pytest.approx(4.0)
    assert payload["hopping_weight"].item() == pytest.approx(6.0)
    # public endpoint telemetry belongs to the main batch only.
    assert payload["active_nodes"].item() == pytest.approx(4.0)
    assert payload["active_edges"].item() == pytest.approx(6.0)


def _single_expert_payload(mt, value):
    outputs = iter([_expert_run_output(mt, _stub(value), 4.0, 6.0)])
    mt._run_one_expert_loss = lambda **kwargs: next(outputs)
    mt.train_lossfunc = _stub(value)
    return mt._build_train_payload(batch_dict=None, batch_info=None, expert_idx=0, range_dis=None)


def test_h11_display_window_expert_metric_averages_only_over_fired_steps():
    """A window with one firing step and one throttled step reports the window
    expert metric averaged over the FIRED step count (2, not diluted to 1); the
    raw active_nodes/active_edges telemetry keeps the plain step-count mean."""
    mt = _minimal_multitrainer()
    mt.distributed_expert = False
    mt._reset_display_window_buffers()

    mt._update_display_window_local(_single_expert_payload(mt, 2.0), current_local_lr=1e-3)
    mt._update_display_window_local(_single_expert_payload(mt, None), current_local_lr=1e-3)

    onsite, hopping, _grad_norm, _lr, active_nodes, active_edges = mt._gather_display_window_expert_metrics()[0]
    assert onsite.item() == pytest.approx(2.0)
    assert hopping.item() == pytest.approx(2.0)
    assert mt._display_window_expert_onsite_steps_local.item() == pytest.approx(1.0)
    assert mt._display_window_expert_hopping_steps_local.item() == pytest.approx(1.0)
    # raw active telemetry: mean over BOTH window steps (unchanged by throttling).
    assert active_nodes.item() == pytest.approx(_ACTIVE_NODES)
    assert active_edges.item() == pytest.approx(_ACTIVE_EDGES)


def test_h11c_payload_metrics_never_source_onsite_from_flow_namespace():
    """The compatible onsite/hopping payload must come from the compatible metric,
    never the flow-namespaced train_flow_* value: the namespaces are disjoint."""
    mt = _minimal_multitrainer()
    flow_only = {
        "train_flow_onsite_loss": torch.tensor(7.0, dtype=mt.dtype),
        "train_flow_hopping_loss": torch.tensor(9.0, dtype=mt.dtype),
    }
    metrics = mt._payload_metrics_from_flow_state(flow_only, prefix="train")
    assert metrics["onsite"].item() != pytest.approx(7.0)
    assert metrics["hopping"].item() != pytest.approx(9.0)
    assert metrics["onsite"].item() == pytest.approx(0.0)
    assert metrics["hopping"].item() == pytest.approx(0.0)
