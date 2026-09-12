#!/usr/bin/env python3
"""Geometry-only inference with a source-bound NACF residual model.

Run from an installed DeePTB checkout (or with PYTHONPATH set to its root).
Checkpoint deserialization is intended for trusted, locally produced models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from ase.io import read

from .inference import load_predictor


def load_cli_predictor(args):
    if args.soc:
        if not args.hopping_checkpoint:
            raise ValueError('compact SOC requires --hopping-checkpoint as well as --checkpoint (onsite)')
        if args.model_backend != 'checkpoint':
            raise ValueError('SOC requires the checkpoint model backend')
        from .spinor_inference import load_soc_predictor
        return load_soc_predictor(args.checkpoint, args.hopping_checkpoint, p2=args.p2, p23=args.p23,
            overlap=args.overlap, soc=args.soc, expected_p2_sha256=args.expected_p2_sha256,
            device=args.device, backend=args.backend, ry_to_ev=args.soc_ry_to_ev,
            onsite_config=args.onsite_config, hopping_config=args.hopping_config,
            p23_missing_policy=args.p23_missing_policy, expected_p23_sha256=args.expected_p23_sha256)
    if args.hopping_checkpoint or args.onsite_config or args.hopping_config:
        raise ValueError('paired checkpoints and their sidecars require --soc')
    return load_predictor(args.checkpoint, args.p2, args.p23, args.overlap,
                          args.expected_p2_sha256, args.device, args.backend,
                          model_backend=args.model_backend, p23_missing_policy=args.p23_missing_policy,
                          expected_p23_sha256=args.expected_p23_sha256)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--p2', required=True)
    parser.add_argument('--p23', required=True)
    parser.add_argument('--overlap', required=True)
    parser.add_argument('--soc', help='Source-bound full spinor projector sidecar')
    parser.add_argument('--expected-p2-sha256', required=True,
                        help='Trusted fingerprint from the checkpoint training config')
    parser.add_argument('--target', required=True, choices=['full_h_minus_nacf'])
    parser.add_argument('--geometry', required=True, help='ASE-readable geometry, possibly multiple frames')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backend', choices=['auto','torch','cuda'], default='auto')
    parser.add_argument('--model-backend', choices=['checkpoint','reference'], default='checkpoint')
    parser.add_argument('--p23-missing-policy', choices=['error','p2_if_missing_pairs'], default='error')
    parser.add_argument('--expected-p23-sha256', help='Trusted training P23 fingerprint; required for composition fallback')
    parser.add_argument('--hopping-checkpoint', help='SOC hopping arm; --checkpoint supplies the onsite arm')
    parser.add_argument('--onsite-config', help='Verified same-run SOC onsite training configuration')
    parser.add_argument('--hopping-config', help='Verified same-run SOC hopping training configuration')
    parser.add_argument('--soc-ry-to-ev', type=float, default=13.605693122994)
    parser.add_argument('--output', required=True, help='New .pt output path; adjacent .json records provenance')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    output = Path(args.output)
    if output.suffix != '.pt':
        parser.error('--output must end in .pt')
    manifest = output.with_suffix('.json')
    if output.exists() or manifest.exists():
        raise FileExistsError('use new output paths')
    predictor = load_cli_predictor(args)
    output_mapper = predictor.full_mapper if args.soc else predictor.idp
    structures = read(args.geometry, index=':')
    if not structures:
        raise ValueError('geometry file contains no structures')
    results, timings = [], []
    for start in range(0, len(structures), args.batch_size):
        if predictor.device.type == 'cuda':
            torch.cuda.synchronize(predictor.device)
        began = time.perf_counter()
        result = predictor(structures[start:start + args.batch_size])
        if predictor.device.type == 'cuda':
            torch.cuda.synchronize(predictor.device)
        timings.append(time.perf_counter() - began)
        results.append({k: v.detach().cpu() for k, v in result.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'schema':'deeptb.nacf_geometry_rme/v1', 'batches':results}, output)
    digest = hashlib.sha256()
    with Path(args.checkpoint).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=digest.hexdigest(),
                  p2_sha256=predictor.bank.p2_manifest_sha256,
                  p23_sha256=predictor.bank.p23.manifest_sha256,
                  overlap_sha256=predictor.bank.overlap.manifest_sha256,
                  soc_sha256=None if predictor.bank.soc is None else predictor.bank.soc.manifest_sha256,
                  has_soc=bool(output_mapper.has_soc),
                  reduced_matrix_element=int(output_mapper.reduced_matrix_element),
                  geometry=str(Path(args.geometry).resolve()), structures=len(structures),
                  target='full_h_minus_nacf', output='absolute Full-H checkpoint RME in eV; S dimensionless',
                  device=str(predictor.device), backend=args.backend, torch_version=torch.__version__,
                  model_dtype=str(predictor.dtype), prior_dtype=str(predictor.bank._anchor.dtype),
                  runtime_overrides=predictor.runtime_model_overrides,
                  p23_missing_policy=predictor.bank.p23_missing_policy,
                  batch_size=args.batch_size, batch_geometry_to_output_seconds=timings,
                  preparation='CPU graph and projector topology; GPU numerical assembly and prediction')
    if args.soc:
        digest = hashlib.sha256()
        with Path(args.hopping_checkpoint).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        report.update(hopping_checkpoint=str(Path(args.hopping_checkpoint).resolve()),
                      hopping_checkpoint_sha256=digest.hexdigest(), soc_ry_to_ev=args.soc_ry_to_ev,
                      onsite_config=args.onsite_config, hopping_config=args.hopping_config,
                      soc_completion='learned uu-real residual; remaining SOC channels from tables')
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
