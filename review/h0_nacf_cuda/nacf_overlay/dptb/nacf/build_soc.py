"""Offline full SOC D sidecar from the exact UPFs bound by a P2 manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .soc import spinor_d_matrix, SOCProjectorStore
from dptb.data.interfaces.p2_table import P2TableStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p2',required=True)
    parser.add_argument('--upf-root',required=True)
    parser.add_argument('--gate1-script',required=True,help='Original qualified UPF compatibility reader')
    parser.add_argument('--species',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from tools.build_nonsoc_p2_tables import _load_gate1
    gate=_load_gate1(Path(args.gate1_script))
    p2=P2TableStore(args.p2)
    output=Path(args.output)
    if output.exists(): raise FileExistsError('use a new immutable SOC sidecar directory')
    output.mkdir(parents=True)
    metadata={}
    for symbol in sorted(set(args.species)):
        source=p2.species[symbol]
        path=Path(args.upf_root)/Path(source['upf_file']).name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['upf_sha256']:
            raise ValueError(f'UPF hash mismatch for {symbol}')
        upf,repair=gate.read_upf_compat(path)
        shells=[int(p.l) for p in upf.projectors]
        js=[p.j for p in upf.projectors]
        if shells!=source['projector_shells'] or bool(upf.has_so)!=source['upf_has_so']:
            raise ValueError(f'UPF projector contract mismatch for {symbol}')
        d=spinor_d_matrix(shells,js,upf.dij_ry,has_so=bool(upf.has_so))
        n=d.shape[0]//2
        if not np.allclose((d[:n,:n]+d[n:,n:])*.5,p2.d_eff(symbol),atol=1e-10,rtol=1e-10):
            raise ValueError(f'SOC normalization does not reproduce scalar table for {symbol}')
        shard=output/(symbol+'.npz')
        np.savez_compressed(shard,d_spinor_ry=d)
        metadata[symbol]=dict(path=shard.name,sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
                              upf_sha256=source['upf_sha256'],projector_shells=shells,projector_j=js,
                              upf_has_so=bool(upf.has_so),upf_metadata_repair=repair)
        print('SOC_BUILT',symbol,d.shape,flush=True)
    manifest=dict(schema='deeptb.soc_projector_table/v1',complete=True,
                  source_p2_manifest_sha256=hashlib.sha256((Path(args.p2)/'manifest.json').read_bytes()).hexdigest(),
                  spin_order='spin_major',harmonic_convention='deeptb_abacus_real',unit='Ry',species=metadata,
                  gate1_script_sha256=hashlib.sha256(Path(args.gate1_script).read_bytes()).hexdigest(),
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    store=SOCProjectorStore(output,p2)
    for symbol in metadata: store.d_spinor(symbol)


if __name__=='__main__': main()
