"""Run a small CPU optimization smoke test from a prior baseline configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dptb.data import _keys
from dptb.nn.build import build_model


def synthetic_data(model, dtype):
    """A three-atom graph with packed AO-product prior features."""
    mapper = model.idp
    atom_type = mapper.chemical_symbol_to_type["C"]
    bond_type = mapper.bond_to_type["C-C"]
    dimension = int(mapper.reduced_matrix_element)
    return {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [-0.3, 1.2, 0.4]], dtype=dtype,
        ),
        _keys.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long,
        ),
        _keys.ATOM_TYPE_KEY: torch.full((3, 1), atom_type, dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.full((6,), bond_type, dtype=torch.long),
        _keys.NODE_H0_KEY: 0.1 * torch.randn(3, dimension, dtype=dtype),
        _keys.EDGE_H0_KEY: 0.1 * torch.randn(6, dimension, dtype=dtype),
    }


def clone_data(data):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}


def smoke(config):
    common = dict(config["common_options"])
    if common["device"] != "cpu":
        raise ValueError("This example runs on CPU")
    seed = common.pop("seed", 2026)
    dtype = getattr(torch, common["dtype"])
    previous_dtype = torch.get_default_dtype()
    # The upstream SO(2) implementation uses the default dtype for allocations.
    torch.set_default_dtype(dtype)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = build_model(
                common_options=common,
                model_options=config["model_options"],
                train_options={},
                no_check=False,
            )
            data = synthetic_data(model, dtype)
            options = config.get("smoke_options", {})
            steps = int(options.get("steps", 20))
            if steps < 1:
                raise ValueError("smoke_options.steps must be positive")
            optimizer = torch.optim.Adam(model.parameters(), lr=float(options.get("learning_rate", 0.003)))
            model.train()

            def objective():
                output = model(clone_data(data))
                # The zero target establishes optimization plumbing only.
                return output[_keys.NODE_FEATURES_KEY].square().mean() + output[_keys.EDGE_FEATURES_KEY].square().mean()

            losses = []
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = objective()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss in baseline smoke test")
                losses.append(float(loss.detach()))
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                    raise RuntimeError("Missing or non-finite gradients in baseline smoke test")
                optimizer.step()
            with torch.no_grad():
                final_loss = float(objective())
            if not final_loss < losses[0]:
                raise RuntimeError("Baseline smoke loss did not decrease")
            return {
                "method": config["model_options"]["embedding"]["method"],
                "device": "cpu",
                "dtype": common["dtype"],
                "steps": steps,
                "initial_loss": losses[0],
                "final_loss": final_loss,
                "losses_before_update": losses,
                "loss_decreased": True,
                "dataset": "synthetic three-atom graph with packed AO H0 inputs and zero targets",
            }
    finally:
        torch.set_default_dtype(previous_dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON smoke receipt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = smoke(json.loads(args.config.read_text(encoding="utf-8")))
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
