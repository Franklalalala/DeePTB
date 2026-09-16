from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.nacf.assembly import NACFTableBank
from dptb.nacf.soc_cpu_reference import CPUScalarAlgorithmSOCAssembler, cpu_soc_blocks
from dptb.tests.test_nacf_gpu import stores


@pytest.mark.parametrize('budget', [0, 64*1024*1024])
def test_cpu_soc_batched_and_scalar_fallback_retain_imaginary_and_spin_flip(budget):
    p2,p23 = stores()
    d = np.array([[2.,.3+.4j],[.3-.4j,3.]])
    soc = SimpleNamespace(manifest={'source_p2_manifest_sha256':None},d_spinor=lambda s:d)
    positions = np.array([[0.,0.,0.],[1.3,.2,.1]])
    symbols,cell = ['X','Y'],np.eye(3)*10
    keys = [(0,0,0,0,0),(1,1,0,0,0),(0,1,0,0,0),(1,0,0,0,0)]
    assembler = CPUScalarAlgorithmSOCAssembler(p2,soc,projector_overlap_cache_max_bytes=budget)
    result = assembler.assemble_sparse_blocks(symbols=symbols,positions_bohr=positions,
                                              cell_bohr=cell,block_keys=keys)
    for i,j,*shift in keys:
        scalar = (p2.onsite_component(symbols[i],'p2_base') if i==j else
            p2.base_component(symbols[i],symbols[j],'p2_base').evaluate(positions[j]-positions[i]))[0,0]
        expected = np.eye(2,dtype=complex)*scalar
        for k,symbol in enumerate(symbols):
            qi = p2.projector(symbol,symbols[i]).evaluate(positions[i]-positions[k])[0,0]
            qj = p2.projector(symbol,symbols[j]).evaluate(positions[j]-positions[k])[0,0]
            expected += qi*d*qj
        np.testing.assert_allclose(result[(i,j,*shift)],expected,atol=1e-13,rtol=1e-13)
        assert np.max(abs(result[(i,j,*shift)].imag)) > 0
    bank = NACFTableBank(p2,p23,soc_store=soc,device='cpu')
    edges,shifts = np.array([[0,1],[1,0]]),np.zeros((2,3),dtype=int)
    reference = cpu_soc_blocks(bank,symbols,positions,cell,edges,shifts)
    actual = bank.prepare(symbols,positions,cell,edges,shifts)()
    for name in reference:
        torch.testing.assert_close(torch.from_numpy(reference[name]),actual[name].to(torch.complex128),
                                   atol=2e-8,rtol=1e-12)
