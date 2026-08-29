#!/usr/bin/env python
"""Launch the band fine-tune with the backbone frozen.

The repo's `freeze` option only covers nnsk models, so the freeze is applied
here by wrapping build_model in the train entrypoint's namespace. Trainable
parameters are printed and asserted non-empty -- a freeze that accidentally
catches everything produces a run that burns GPU hours and learns nothing,
and nothing downstream would report it.
"""
import argparse
import sys

REPO = "/data/wgh/0828_band_finetune/code/DeePTB_bandft"
sys.path.insert(0, REPO)

import importlib  # noqa: E402
import torch  # noqa: E402

# `dptb.entrypoints.__init__` rebinds the name `train` to the function, so the
# attribute lookup would hand back the function instead of the module.
importlib.import_module("dptb.entrypoints.train")
train_mod = sys.modules["dptb.entrypoints.train"]

# Everything outside these name fragments is frozen. In this architecture every
# parameter lives under `embedding`; the RME output head is out_node/out_edge
# (plus their element tensor-product partners), which is what the startup log
# calls "Output Head: route=legacy_rme". Those are the smallest set that can
# still re-aim the model at a band target while the 3-layer backbone holds.
TRAINABLE_PATTERNS = ("embedding.out_node", "embedding.out_edge")


def freeze_backbone(model, patterns, verbose=True):
    n_train = n_frozen = 0
    trainable_names = []
    for name, p in model.named_parameters():
        if any(pat in name for pat in patterns):
            p.requires_grad_(True)
            n_train += p.numel()
            trainable_names.append(name)
        else:
            p.requires_grad_(False)
            n_frozen += p.numel()
    total = n_train + n_frozen
    if verbose:
        print("=" * 70)
        print("FREEZE: trainable %d / %d params (%.3f%%), frozen %d"
              % (n_train, total, 100.0 * n_train / max(total, 1), n_frozen))
        for nm in trainable_names[:12]:
            print("   trainable:", nm)
        if len(trainable_names) > 12:
            print("   ... and %d more" % (len(trainable_names) - 12))
        print("=" * 70, flush=True)
    if n_train == 0:
        raise SystemExit(
            "freeze left zero trainable parameters -- patterns %r match nothing "
            "in this model. Check the parameter names printed above." % (patterns,))
    if n_frozen == 0:
        raise SystemExit(
            "freeze left nothing frozen -- the patterns matched every parameter, "
            "so this would be a full fine-tune, not a head fine-tune.")
    return n_train, n_frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--init-model", required=True)
    ap.add_argument("--log", default="log.txt")
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--list-params", action="store_true",
                    help="build the model, print parameter names, and exit")
    args = ap.parse_args()

    _orig_build = train_mod.build_model

    def build_model_frozen(*a, **kw):
        model = _orig_build(*a, **kw)
        if args.list_params:
            for name, p in model.named_parameters():
                print("PARAM %-72s %s" % (name, tuple(p.shape)))
            raise SystemExit(0)
        if not args.no_freeze:
            freeze_backbone(model, TRAINABLE_PATTERNS)
        else:
            print("FREEZE: disabled, full fine-tune", flush=True)
        return model

    train_mod.build_model = build_model_frozen

    train_mod.train(
        INPUT=args.input,
        init_model=args.init_model,
        restart=None,
        train_soc=False,
        output=args.output,
        log_level=20,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
