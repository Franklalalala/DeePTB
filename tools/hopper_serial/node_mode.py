import os,sys,json,subprocess
from pathlib import Path
r=Path('/scratch/sheng.lei/0912_h0_serial_edgemoe');job=os.environ['PBS_JOBID'].split('.')[0]
e=json.loads((r/job/'new_runtime.json').read_text());os.environ.update(e)
node=Path('/tmp/sheng.lei/h0_serial_'+job);node.mkdir(parents=True,exist_ok=True)
os.environ.update(PYTHONPATH=str(r/'runtime')+':'+str(r/'DeePTB')+':/scratch/Projects/CFP-04/CFP04-CF-019/p23_h0res_shared/SO2CUDA/src:/scratch/Projects/CFP-04/CFP04-CF-019/p23_h0res_shared/SO2CUDA',DPTB_TARGET_KIND='h0res',DPTB_SO2_FUSION_MODE='streamed_m_major_fused_p0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TEMP=str(node),TMP=str(node),TMPDIR=str(node),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
for k in ['NCCL_ASYNC_ERROR_HANDLING','DPTB_RESTART','DPTB_INIT_MODEL','DPTB_INPUT','DPTB_OUTPUT','DPTB_SO2_FUSE_M_CUBLAS']:os.environ.pop(k,None)
py=str(node/'env/bin/python')
mode=sys.argv[1]
if mode=='tests':
 os.chdir(r/'DeePTB')
 cmd=[py,'-m','pytest','-q','-p','no:cacheprovider','dptb/tests/test_multi_train_max_steps.py','dptb/tests/test_serial_edge_prior_2b.py','dptb/tests/test_distance_expert_mask.py','dptb/tests/test_lem_moe_v3_prior_2b.py','dptb/tests/test_prior2b_pa.py','--basetemp',str(node/'pytest')]
else:cmd=[py,str(r/'smoke_contract.py'),mode]
raise SystemExit(subprocess.call(cmd))
