import torch
from contextlib import contextmanager

from dptb.data.dataloader import (
    AtomicDataCostEstimator,
    DataLoader,
    resolve_dynamic_batch_options,
    split_batch_for_oom,
)
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.utils.torch_geometric import Data


class ToyDataset:
    def __init__(self, node_counts):
        self.node_counts = list(node_counts)

    def __len__(self):
        return len(self.node_counts)

    def __getitem__(self, idx):
        n = int(self.node_counts[idx])
        return Data(
            pos=torch.zeros((n, 3), dtype=torch.float32),
            edge_index=torch.zeros((2, n + 1), dtype=torch.long),
            env_index=torch.zeros((2, n + 2), dtype=torch.long),
            onsitenv_index=torch.zeros((2, n + 3), dtype=torch.long),
            kpoint=torch.zeros((2, 3), dtype=torch.float32),
            eigenvalue=torch.zeros((1, 4), dtype=torch.float32),
        )


def test_cost_estimator_uses_deeptb_graph_terms():
    data = ToyDataset([4])[0]
    estimator = AtomicDataCostEstimator(
        mode="cost",
        cost_weights={
            "graph": 10.0,
            "node": 2.0,
            "edge": 3.0,
            "env": 5.0,
            "onsitenv": 7.0,
            "kpoint": 11.0,
            "eig_band_square": 0.0,
        },
    )

    parts = estimator.parts(data)

    assert parts["graph"] == 1
    assert parts["node"] == 4
    assert parts["edge"] == 5
    assert parts["env"] == 6
    assert parts["onsitenv"] == 7
    assert parts["kpoint"] == 2
    assert estimator(data) == 10 + 2 * 4 + 3 * 5 + 5 * 6 + 7 * 7 + 11 * 2


def test_dynamic_loader_caps_cost_and_keeps_batch_size_as_max_samples():
    dataset = ToyDataset([4, 5, 12, 3])
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "mode": "node",
            "max_cost": 10,
        },
    )

    batches = list(loader)

    assert [b.__dptb_sample_indices__ for b in batches] == [[0, 1], [2], [3]]
    assert [b.num_graphs for b in batches] == [2, 1, 1]
    assert batches[0].__dptb_batch_cost__ == 9
    assert batches[0].__dptb_batch_num_nodes__ == 9
    assert batches[0].__dptb_batch_max_item_cost__ == 5
    assert batches[1].__dptb_batch_cost__ == 12


def test_calibration_derives_quantile_from_fixed_batch_totals():
    dataset = ToyDataset([1, 2, 10, 20, 5])

    opts = resolve_dynamic_batch_options(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "mode": "node",
            "calibrate": True,
            "calibration_batches": 10,
            "calibration_quantile": 0.5,
        },
    )

    assert opts["max_cost"] == 5
    assert opts["max_samples"] == 2
    assert opts["calibrated"] is True
    assert opts["calibration_batch_costs"] == [3, 30, 5]


def test_split_batch_for_oom_bisects_and_preserves_metadata():
    dataset = ToyDataset([2, 3, 4, 5])
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "mode": "node",
            "max_cost": 100,
        },
    )
    batch = next(iter(loader))

    left, right = split_batch_for_oom(batch)

    assert left.__dptb_sample_indices__ == [0, 1]
    assert right.__dptb_sample_indices__ == [2, 3]
    assert left.__dptb_batch_cost__ == 5
    assert right.__dptb_batch_cost__ == 9
    assert left.num_graphs == 2
    assert right.num_graphs == 2


class _DummyTagger:
    @contextmanager
    def tag(self, *args, **kwargs):
        yield

    def dump_cuda_mem_summary(self, where):
        return None


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


def test_multitrainer_microbatch_fallback_accumulates_one_optimizer_step():
    dataset = ToyDataset([2, 3, 4, 5])
    batch = next(iter(DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        dynamic_batch={"enabled": True, "mode": "node", "max_cost": 100},
    )))

    trainer = MultiTrainer.__new__(MultiTrainer)
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = _CountingSGD([param])
    observed_states = []

    trainer.distributed_expert = False
    trainer.distributed_rank0_prepare_batch = False
    trainer.distance_ranges = [(0.0, 1.0)]
    trainer.num_experts = 1
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer.iter = 7
    trainer.clip_grad_norm = 1000.0
    trainer.optimizers = [opt]
    trainer.train_lossfunc = object()
    trainer.dynamic_batch_oom_shrink_factor = 0.8
    trainer.model = torch.nn.Linear(1, 1)
    trainer._tagger = _DummyTagger()
    trainer.train_loader = type(
        "Loader",
        (),
        {
            "batch_sampler": type("Sampler", (), {"max_cost": 100})(),
            "dynamic_batch_options": {"max_cost": 100},
        },
    )()

    trainer._is_cuda_device = lambda: False
    trainer._reset_cuda_memory_peak = lambda: None
    trainer._expert_parameters = lambda expert_idx: [param]
    trainer._prepare_batch_bundle = lambda micro_batch, with_lengths=True: (
        {"cost": torch.tensor(float(micro_batch.__dptb_batch_cost__))},
        {},
    )

    def _build_train_payload(batch_dict, batch_info, expert_idx, range_dis):
        loss = param * batch_dict["cost"]
        return {
            "loss": loss,
            "expert_onsite": loss.detach(),
            "expert_hopping": loss.detach(),
            "onsite_weighted_sum": loss.detach(),
            "hopping_weighted_sum": loss.detach(),
            "active_nodes": torch.tensor(1.0),
            "active_edges": torch.tensor(1.0),
            "onsite_l1_sum": None,
            "onsite_mse_sum": None,
            "onsite_cnt": None,
            "hopping_l1_sum": None,
            "hopping_mse_sum": None,
            "hopping_cnt": None,
            "z_values": [],
            "load_cv_values": [],
        }

    trainer._build_train_payload = _build_train_payload
    trainer._to_float_scalar = lambda x: float(x.detach().item() if torch.is_tensor(x) else x)
    trainer._to_int_scalar = lambda x: int(x.detach().item() if torch.is_tensor(x) else x)
    trainer._compute_stitched_loss_by_reduce = lambda payloads, criterion: None
    trainer._local_scheduler_step = lambda metric: None
    trainer._add_cuda_memory_state = lambda state, metrics: None
    trainer._gather_cuda_memory_metrics = lambda: {}
    trainer.call_plugins = lambda queue_name, time, **state: observed_states.append(state)

    trainer._run_single_process_microbatch_fallback(batch)

    assert opt.step_calls == 1
    assert trainer.iter == 8
    assert trainer.train_loader.batch_sampler.max_cost == 80
    assert observed_states[0]["dynamic_batch_oom_fallback"] == 1
    assert observed_states[0]["dynamic_batch_microbatches"] == 2
    assert observed_states[0]["batch_cost"] == 14
