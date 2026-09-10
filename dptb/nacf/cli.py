#!/usr/bin/env python3
"""Geometry-only non-SOC inference with a source-bound NACF residual model.

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



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--p2', required=True)
    parser.add_argument('--p23', required=True)
    parser.add_argument('--overlap', required=True)
    parser.add_argument('--expected-p2-sha256', required=True,
                        help='Trusted fingerprint from the checkpoint training config')
    parser.add_argument('--target', required=True, choices=['full_h_minus_nacf'])
    parser.add_argument('--geometry', required=True, help='ASE-readable geometry, possibly multiple frames')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backend', choices=['auto','torch','cuda'], default='auto')
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
    predictor = load_predictor(args.checkpoint, args.p2, args.p23, args.overlap,
                               args.expected_p2_sha256, args.device, args.backend)
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
                  geometry=str(Path(args.geometry).resolve()), structures=len(structures),
                  target='full_h_minus_nacf', output='absolute Full-H triangular non-SOC RME in eV; S dimensionless',
                  device=str(predictor.device), backend=args.backend, torch_version=torch.__version__,
                  model_dtype=str(predictor.dtype), prior_dtype=str(predictor.bank._anchor.dtype),
                  runtime_overrides={'so2_fusion_mode':'streamed_m_major_ref','mole_linear_mode':'split_loop'},
                  batch_size=args.batch_size, batch_geometry_to_output_seconds=timings,
                  preparation='CPU graph and projector topology; GPU numerical assembly and prediction')
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
