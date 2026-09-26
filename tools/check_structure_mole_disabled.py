#!/usr/bin/env python
"""Cross-checkout tensor comparison, following the r12b disabled-head probe.

Use --collect PATH under each checkout's PYTHONPATH, then --compare OLD NEW.
Both absent and explicit disabled options are tested for dense and small MoE.
"""
import argparse
import copy
import json

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.compare:
        old, new = [torch.load(path, map_location="cpu", weights_only=False) for path in args.compare]
        assert old.keys() == new.keys()
        counts = {}
        for case in old:
            counts[case] = {}
            for kind in old[case]:
                assert old[case][kind].keys() == new[case][kind].keys()
                for key, a in old[case][kind].items():
                    b = new[case][kind][key]
                    equal = a is None and b is None
                    if a is not None and b is not None:
                        equal = (a.shape == b.shape and a.dtype == b.dtype and torch.equal(
                            a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))
                    assert equal, (case,kind,key)
                counts[case][kind] = len(old[case][kind])
        print(json.dumps(dict(exact=True,counts=counts),indent=2))
        return
    assert args.collect, "specify --collect or --compare"
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "requested CUDA disabled-path probe must execute on GPU"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    import dptb
    from dptb.tests.shift_head_helpers import config, batch
    from dptb.nn.build import build_model
    print("import", dptb.__file__, flush=True)
    result = {}
    for moe in (False, True):
        for explicit in (False, True):
            torch.manual_seed(9414)
            cfg = config(moe=moe, device=args.device)
            if explicit:
                cfg["model_options"]["embedding"]["structure_mole"] = {"enabled":False}
            model = build_model(**cfg)
            out = model(copy.deepcopy(batch(model)))
            (out["node_features"].square().sum()+out["edge_features"].square().sum()).backward()
            result[f"{moe}/{explicit}"] = dict(
                output={k:v.detach().cpu().clone() for k,v in out.items() if torch.is_tensor(v)},
                grad={k:p.grad.detach().cpu().clone() if p.grad is not None else None for k,p in model.named_parameters()},
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                rng={"cpu":torch.get_rng_state(), **({"cuda":torch.cuda.get_rng_state()} if args.device.startswith("cuda") else {})})
    torch.save(result,args.collect)


if __name__ == "__main__":
    main()
