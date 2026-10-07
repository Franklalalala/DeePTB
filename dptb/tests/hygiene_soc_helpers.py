"""Opt-in real SOC fixture shared with the standalone compatibility receipt."""
import copy
import json
from pathlib import Path

import torch

from dptb.data import AtomicData
from dptb.data.AtomicDataDict import with_edge_vectors
from dptb.data.build import build_dataset
from dptb.data.dataloader import Collater
from dptb.utils.argcheck import collect_cutoffs, normalize


def reference_config(path, dtype):
    options = normalize(json.loads(Path(path).read_text()))
    options["common_options"].update(device="cpu", dtype=dtype)
    # Only execution backends change; keep the base_on basis, SOC contract,
    # architecture and H0 settings. No model.double() after construction.
    options["model_options"]["embedding"].update(
        so2_fusion_mode="staged", mole_linear_mode="split_loop")
    return options


def reference_data(options, index=8):
    dataset = build_dataset(**collect_cutoffs(options), **options["data_options"]["validation"],
                            **options["common_options"])
    batch = Collater()([dataset[index]])
    data = AtomicData.to_AtomicDataDict(batch)
    dtype = getattr(torch, options["common_options"]["dtype"])
    data = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else copy.deepcopy(v)
            for k, v in data.items()}
    return with_edge_vectors(data, with_lengths=True)
