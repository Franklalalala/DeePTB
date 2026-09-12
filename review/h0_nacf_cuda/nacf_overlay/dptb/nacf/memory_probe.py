"""Isolate extension import, first native launch, resident table and output memory.

Run in a fresh process. NVIDIA process MiB includes CUDA runtime/module memory;
PyTorch allocated bytes count live tensors; reserved bytes include its allocator
cache. MiB readings have driver granularity and are not exact cubin byte counts.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from dptb.data.interfaces.p2_table import P2TableStore
from ._cuda import extension, evaluate
from .radial import TorchRadialBlockTable
from .production_benchmark import memory_snapshot, storage_bytes


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p2',required=True)
    parser.add_argument('--species',default='Si')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    device=torch.device(args.device)
    torch.cuda.init()
    # Release tensor allocator cache identically before every process snapshot.
    def snapshot():
        gc.collect(); torch.cuda.empty_cache()
        return memory_snapshot(device)
    stages={'context':snapshot()}
    extension()
    stages['extension_imported_before_launch']=snapshot()
    source=P2TableStore(args.p2).base_component(args.species,args.species,'p2_base')
    table=TorchRadialBlockTable(source,device=device,backend='cuda')
    vectors=torch.tensor([[0.,0.,1.]],dtype=torch.float64,device=device)
    stages['table_and_input_loaded']=snapshot()
    torch.cuda.reset_peak_memory_stats(device)
    value=evaluate(table,vectors)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(value.cpu(),torch.as_tensor(source.evaluate([0.,0.,1.]))[None],atol=1e-9,rtol=1e-9)
    peak=torch.cuda.max_memory_allocated(device)
    output_bytes=value.untyped_storage().nbytes()
    del value
    stages['first_native_launch_output_freed']=snapshot()
    value=evaluate(table,vectors)
    torch.cuda.synchronize(device)
    del value
    stages['second_native_launch_output_freed']=snapshot()
    report=dict(scope=__doc__,gpu=torch.cuda.get_device_name(device),torch_version=torch.__version__,
                species=args.species,table_bytes=storage_bytes(table.buffers()),input_bytes=storage_bytes([vectors]),
                output_bytes=output_bytes,first_launch_peak_allocated_bytes=peak,stages=stages)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
