"""Higher-LR whole-stack fine-tuning with edge-budget GPU batches.

Legacy checkpoints preserve weights/AdamW moments but start a newly declared
sampling phase. New checkpoints additionally record the committed data cursor.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from dptb.data import AtomicData, AtomicDataDict as A
from dptb.data.build import build_dataset
from dptb.data.dataloader import DataLoader, AtomicDataCostEstimator
from dptb.data.transforms import OrbitalMapper
from dptb.nn.build import build_model
from dptb.nnops.loopscf.stack import install_stack_loop, adaptive_objective
from dptb.nnops.loopscf.dynamic import CostController, backward_with_retry, edge_budget_batches
from dptb.utils.argcheck import normalize
from dptb.utils.dpa4_optim import WarmupStableDecayLR
from examples.loopscf.anneal_stack import atomic_json, losses_for
from examples.loopscf.evaluate import file_sha256


def identity_collate(items):
    return items


class CommittedBatches:
    """Prefetch cannot advance this cursor; only a successful update commits it."""
    def __init__(self, indices, size, seed, state=None):
        self.indices, self.size = list(indices), size
        self.rng = torch.Generator().manual_seed(seed)
        self.epoch, self.cursor = 0, 0
        self.order = self._order()
        if state:
            self.order, self.cursor, self.epoch = state['order'], state['cursor'], state['epoch']
            self.rng.set_state(state['rng'])

    def _order(self):
        return [self.indices[i] for i in torch.randperm(len(self.indices), generator=self.rng).tolist()]

    def __iter__(self):
        for start in range(self.cursor, len(self.order), self.size):
            yield self.order[start:start+self.size]

    def commit(self, count):
        self.cursor += count
        if self.cursor > len(self.order):
            raise RuntimeError('sampler cursor overflow')

    def next_epoch(self):
        if self.cursor != len(self.order):
            raise RuntimeError('uncommitted data remain')
        self.epoch += 1
        self.cursor, self.order = 0, self._order()

    def state_dict(self):
        return dict(order=self.order, cursor=self.cursor, epoch=self.epoch, rng=self.rng.get_state())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ['input', 'base-model', 'output']:
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--resume')
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--initial-graphs', '--initial-microbatch', dest='initial_microbatch', type=int, default=4,
                    help='number of initial CPU graphs used to calibrate the edge budget')
    ap.add_argument('--max-graphs', '--max-microbatch', dest='max_microbatch', type=int, default=8)
    ap.add_argument('--memory-target-gib', type=float, default=68)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--bridge-lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--schedule', choices=['cosine', 'wsd'], default='cosine')
    ap.add_argument('--schedule-clock', choices=['deadline', 'steps'], default='deadline')
    ap.add_argument('--total-steps', type=int,
                    help='total committed updates including the resume parent; steps clock only')
    ap.add_argument('--warmup-lr', type=float, default=1e-6)
    ap.add_argument('--min-lr', type=float, default=1e-6)
    ap.add_argument('--decay-ratio', type=float, default=.65,
                    help='WSD progress fraction at which final cosine decay starts')
    ap.add_argument('--hours', type=float, default=8)
    ap.add_argument('--deadline', type=float)
    ap.add_argument('--calibration-size', type=int, default=512)
    ap.add_argument('--gate-calibration', type=int, default=100)
    ap.add_argument('--validation-limit', type=int, default=400)
    ap.add_argument('--initial-validation-limit', type=int, default=None,
                    help='optional smaller initial probe; final validation uses validation-limit')
    ap.add_argument('--checkpoint-every', type=int, default=100)
    ap.add_argument('--keep-checkpoints', type=int, default=3)
    ap.add_argument('--stop-after', type=int, default=0, help='diagnostic update cap without compressing the LR schedule')
    ap.add_argument('--inject-oom-once', action='store_true', help='smoke-only retry verification')
    args = ap.parse_args()
    if args.schedule_clock == 'steps' and (args.schedule != 'wsd' or args.deadline is not None):
        ap.error('steps clock requires WSD and no wall-clock deadline')
    if args.total_steps is not None and (args.schedule_clock != 'steps' or args.total_steps < 1):
        ap.error('total-steps requires a positive total with steps clock')
    if args.keep_checkpoints < 1:
        ap.error('keep-checkpoints must be positive')
    schedule_total=args.total_steps or args.steps
    update_limit=min(args.steps,args.stop_after) if args.stop_after>0 else args.steps
    if min(args.K,args.steps,args.batch_size,args.initial_microbatch,args.max_microbatch,args.warmup,args.checkpoint_every)<1 or min(args.lr,args.bridge_lr,args.hours,args.memory_target_gib)<=0:
        ap.error('positive budgets required')
    if args.schedule == 'wsd':
        if not (0 <= args.warmup_lr <= min(args.lr,args.bridge_lr) and
                0 <= args.min_lr <= min(args.lr,args.bridge_lr) and
                0 < args.decay_ratio < 1 and
                args.warmup < round(schedule_total*args.decay_ratio) < schedule_total):
            ap.error('invalid WSD phase boundaries or LR bounds')
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():
        raise RuntimeError('use a fresh revision directory under the original run')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    begin=time.time()
    deadline=None if args.schedule_clock == 'steps' else args.deadline or begin+args.hours*3600
    if deadline is not None and deadline <= begin+60:
        raise ValueError('insufficient remaining wall-clock budget')
    cfg=normalize(json.loads(Path(args.input).read_text()))
    common=dict(cfg['common_options'],device='cuda:0'); data=cfg['data_options']
    datasets={s:build_dataset(**data[s],r_max=data.get('r_max'),er_max=data.get('er_max'),oer_max=data.get('oer_max'),**common) for s in ['train','validation']}
    if not 0 <= args.calibration_size < len(datasets['train']):
        raise ValueError('invalid calibration partition')
    partition=np.random.default_rng(20260910).permutation(len(datasets['train'])).tolist()
    cal_indices, train_indices=partition[:args.calibration_size],partition[args.calibration_size:]
    atomic_json(root/'partition.json',dict(train=train_indices,gate_calibration=cal_indices))
    model=build_model(checkpoint=args.base_model,model_options=cfg['model_options'],common_options=common).cuda()
    idp=OrbitalMapper(common['basis'],method='e3tb',device='cuda:0')
    torch.manual_seed(args.seed)
    install_stack_loop(model,idp,K=args.K,strategy='bptt')
    groups={g:[] for g in ['body','bridge','gate']}
    for name,p in model.named_parameters():
        if p.requires_grad:
            groups['bridge' if '.stack_bridge.' in name else 'gate' if '.stack_exit.' in name else 'body'].append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,lr=args.bridge_lr if g=='bridge' else args.lr,group_name=g) for g,ps in groups.items()],weight_decay=.01,foreach=False)
    parent_step=0; parent_cumulative_step=0; resume_state=None
    if args.resume:
        raw=torch.load(args.resume,map_location='cpu',weights_only=False)
        saved=raw['stack_protocol']
        if saved['K']!=args.K or saved.get('strategy','bptt')!='bptt' or raw['stage']!='joint':
            raise ValueError('resume requires a joint-stage checkpoint at matching depth/strategy')
        for key in ['seed','calibration_size']:
            if saved.get(key)!=getattr(args,key):
                raise ValueError('resume changes '+key)
        if raw['config']['model_options']!=cfg['model_options'] or raw['config']['data_options']!=cfg['data_options']:
            raise ValueError('resume model/data configuration differs')
        model.load_state_dict(raw['model_state_dict'],strict=True)
        optimizer.load_state_dict(raw['optimizer_state_dict'])
        parent_step=raw['step']
        parent_cumulative_step=raw.get('cumulative_step',parent_step)
        resume_state=raw.get('dynamic_state')
        if resume_state:
            if saved['batch_size']!=args.batch_size:
                raise ValueError('dynamic resume requires same logical batch size')
            torch.set_rng_state(raw['torch_rng_state']);torch.cuda.set_rng_state_all(raw['cuda_rng_state'])
            random.setstate(resume_state['python_rng']);np.random.set_state(resume_state['numpy_rng'])
        del raw
    # load_state_dict also restores old optimizer hyperparameters; explicitly override.
    resume_group_lrs=[float(g['lr']) for g in optimizer.param_groups]
    for g,name in zip(optimizer.param_groups,groups):
        g['group_name']=name;g['foreach']=False
        g['peak_lr']=args.bridge_lr if name=='bridge' else args.lr
        g['warmup_start_lr']=min(g['peak_lr'],float(g['lr'])) if args.resume else .03*g['peak_lr']
    if args.schedule_clock == 'steps':
        remaining=schedule_total-parent_cumulative_step
        if remaining <= 0:
            raise ValueError('resume checkpoint already reached the requested total updates')
        update_limit=min(update_limit,remaining)
    # A parent already on the same plateau must not repeat warmup after changing
    # the horizon. Its optimizer moments and committed sampler are preserved.
    warmup_completed=bool(args.resume and all(math.isclose(lr,g['peak_lr'],rel_tol=1e-6)
                          for lr,g in zip(resume_group_lrs,optimizer.param_groups)))
    wsd = None
    if args.schedule == 'wsd':
        for g in optimizer.param_groups:
            g['lr'] = g['peak_lr']
            g['initial_lr'] = g['peak_lr']
            g['warmup_start_lr'] = args.warmup_lr
        wsd = WarmupStableDecayLR(optimizer,total_steps=schedule_total,warmup_steps=args.warmup,
            warmup_lr=args.warmup_lr,min_lr=args.min_lr,decay_ratio=args.decay_ratio)
    sampler=CommittedBatches(train_indices,args.batch_size,args.seed+10000,resume_state['sampler'] if resume_state else None)
    target=min(int(args.memory_target_gib*1024**3),torch.cuda.get_device_properties(0).total_memory-6*1024**3)
    controller=CostController(**resume_state['controller']) if resume_state else CostController(1,args.max_microbatch,target)
    controller.max_graphs=args.max_microbatch; controller.target_bytes=target
    initialized=resume_state is not None
    metadata=dict(vars(args),deadline_epoch=deadline,started_epoch=begin,parent_step=parent_step,parent_cumulative_step=parent_cumulative_step,
        resume_semantics=('strict weights+AdamW; legacy parent starts new seeded sampling phase; dynamic parent restores committed cursor'
                          if args.resume else 'pretrained base only; fresh loop adapters, AdamW, sampler and LR schedule'),
        base_checkpoint_sha256=file_sha256(args.base_model),
        parent_checkpoint_sha256=file_sha256(args.resume) if args.resume else None,
        train_count=len(train_indices),calibration_count=len(cal_indices),gpu=torch.cuda.get_device_name(),
        trainable_parameters={g:sum(p.numel() for p in ps) for g,ps in groups.items()},
        strategy='bptt',max_graphs_per_update=args.max_microbatch,target_bytes=target,
        batch_policy='edge-budget pack then direct whole-batch update; only CUDA OOM activates weighted split retry',
        mutable_buffer_snapshot_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        warmup_start_lrs={g['group_name']:g['warmup_start_lr'] for g in optimizer.param_groups},
        optimizer='AdamW',
        schedule_total_steps=schedule_total,planned_new_updates=update_limit,
        resume_group_lrs=resume_group_lrs,resume_warmup_completed=warmup_completed,
        lr_clock=('cumulative committed optimizer updates only; no time-based training stop'
                  if args.schedule_clock == 'steps' else
                  'max(committed updates, elapsed training fraction * total steps); clock starts after initial validation'
                  if wsd else 'legacy max(update fraction, elapsed runner fraction)'),
        loss='graph mean: 0.8*expected_H+0.2*mean_depth_H-0.0005*entropy',
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[2])):file_sha256(p) for p in [Path(__file__),Path(sys.modules[CostController.__module__].__file__),Path(sys.modules[install_stack_loop.__module__].__file__)]})
    atomic_json(root/'protocol.json',metadata);atomic_json(root/'train_config.json',cfg)
    print('PROTOCOL',json.dumps(metadata),flush=True)
    exclude=[A.KPOINT_KEY,A.ENERGY_EIGENVALUE_KEY]
    step=0; stopping=[False]; estimator=AtomicDataCostEstimator('edge')
    schedule_step=parent_cumulative_step
    signal.signal(signal.SIGUSR1,lambda *unused:stopping.__setitem__(0,True))

    def save(name,stage='joint'):
        tmp=root/(name+'.tmp')
        torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            step=step,parent_step=parent_step,cumulative_step=parent_cumulative_step+step,stage=stage,stack_protocol={**vars(args),'strategy':'bptt'},config=cfg,
            schedule_state=dict(clock=args.schedule_clock,step=schedule_step,total_steps=schedule_total),
            torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
            dynamic_state=dict(sampler=sampler.state_dict(),controller=controller.state_dict(),
                               python_rng=random.getstate(),numpy_rng=np.random.get_state())),tmp)
        tmp.replace(root/(name+'.pth'))
        if name.startswith('step_'):
            for old in sorted(root.glob('step_*.pth'))[:-args.keep_checkpoints]:old.unlink()

    @torch.no_grad()
    def validate(tag):
        rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all()
        model.eval();rows=[]
        limit=args.initial_validation_limit if tag=='initial' and args.initial_validation_limit is not None else args.validation_limit
        for index in range(min(limit,len(datasets['validation']))):
            batch=next(iter(DataLoader(dataset=[datasets['validation'][index]],batch_size=1,exclude_keys=exclude)))
            ref=AtomicData.to_AtomicDataDict(batch.cuda());out=model(dict(ref));ls=losses_for(out,ref,idp)
            rows.append(dict(index=index,per_step_ev=ls[0].tolist(),p=out['_exit_probabilities'][0].tolist()))
        atomic_json(root/('validation_'+tag+'.json'),dict(step=step,results=rows,
            mean_per_step_ev=np.mean([r['per_step_ev'] for r in rows],axis=0).tolist() if rows else []))
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng)
        model.train()

    validate('initial')
    schedule_begin=time.time()
    metadata['schedule_started_epoch']=schedule_begin
    atomic_json(root/'protocol.json',metadata)
    injected=[False]
    with (root/'history.jsonl').open('a',buffering=1) as history:
        while step<update_limit and (deadline is None or time.time()<deadline) and not stopping[0]:
            # Dataset returns CPU records. Prefetch uses a separate RNG and cannot commit sampler state.
            loader=torch.utils.data.DataLoader(datasets['train'],batch_sampler=sampler,num_workers=1,
                collate_fn=identity_collate,generator=torch.Generator().manual_seed(args.seed+sampler.epoch),pin_memory=False)
            accepted_batches=edge_budget_batches(loader,controller,estimator,args.initial_microbatch,not initialized)
            for items in accepted_batches:
                if step>=update_limit or (deadline is not None and time.time()>=deadline) or stopping[0]:break
                tick=time.monotonic();model.train()
                indices=sampler.order[sampler.cursor:sampler.cursor+len(items)]
                costs=[estimator(x) for x in items]
                initialized=True
                if wsd:
                    # Reuse the established WSD curve, with a virtual step clock
                    # so a slow run still completes the final decay by deadline.
                    if args.schedule_clock == 'steps':
                        schedule_step=parent_cumulative_step+step
                        if warmup_completed:schedule_step=max(schedule_step,args.warmup)
                    else:
                        elapsed_fraction=(time.time()-schedule_begin)/max(1.,deadline-schedule_begin)
                        schedule_step=min(schedule_total,max(step,int(elapsed_fraction*schedule_total)))
                    for g,lr in zip(optimizer.param_groups,wsd.get_lr_at_step(schedule_step)):
                        g['lr']=lr
                    lr_phase=('warmup' if schedule_step<wsd.warmup_steps else
                              'stable' if schedule_step<wsd.decay_start_step else 'decay')
                else:
                    progress=max((step+1)/args.steps,(time.time()-begin)/(deadline-begin))
                    schedule_step=step
                    lr_phase='cosine'
                    warm=min(1.,(step+1)/args.warmup)
                    decay=.03+.97*.5*(1+math.cos(math.pi*min(1.,progress)))
                    for g in optimizer.param_groups:
                        g['lr']=(g['warmup_start_lr']+(g['peak_lr']-g['warmup_start_lr'])*warm)*decay
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                static=torch.cuda.memory_allocated()

                def capture():
                    return (torch.get_rng_state(),torch.cuda.get_rng_state_all(),
                            {n:b.detach().clone() for n,b in model.named_buffers()})
                def restore(state):
                    torch.set_rng_state(state[0]);torch.cuda.set_rng_state_all(state[1])
                    with torch.no_grad():
                        for name,b in model.named_buffers():b.copy_(state[2][name])
                def micro_backward(local_indices,weight):
                    batch=next(iter(DataLoader(dataset=[items[i] for i in local_indices],batch_size=len(local_indices),exclude_keys=exclude)))
                    ref=AtomicData.to_AtomicDataDict(batch.cuda());out=model(dict(ref));ls=losses_for(out,ref,idp)
                    expected,ent=adaptive_objective(ls,out['_exit_probabilities'],.0005)
                    loss=.8*expected+.2*ls.mean()-.2*.0005*ent.mean()
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                    (loss*weight).backward();torch.cuda.synchronize()
                    if args.inject_oom_once and not injected[0]:
                        injected[0]=True
                        raise torch.cuda.OutOfMemoryError('injected smoke OOM after real backward')
                    return dict(loss=float(loss.detach()),depth=ls.detach().mean(0).tolist(),
                        p=out['_exit_probabilities'].detach().mean(0).tolist(),weight=weight,counts=out['_stack_counts'])
                def retry(local_indices,message,attempt):
                    print('CUDA_OOM_RETRY',json.dumps(dict(step=step+1,sample_indices=[indices[i] for i in local_indices],attempt=attempt,error=message[:500])),flush=True)
                results,micros,retries=backward_with_retry(costs,controller,micro_backward,
                    lambda:optimizer.zero_grad(set_to_none=True),capture,restore,on_retry=retry)
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                torch.cuda.synchronize()
                # NO RETRY around commit: partial optimizer mutations must fail closed.
                optimizer.step();torch.cuda.synchronize()
                sampler.commit(len(items));step+=1
                peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
                used=max(sum(costs[i] for i in m) for m in micros)
                controller.observe(used,max(peak,reserved),static)
                free,total=torch.cuda.mem_get_info()
                row=dict(step=step,parent_step=parent_step,cumulative_step=parent_cumulative_step+step,seconds=time.monotonic()-tick,elapsed=time.time()-begin,
                    lr=optimizer.param_groups[0]['lr'],bridge_lr=optimizer.param_groups[1]['lr'],
                    schedule=args.schedule,schedule_step=schedule_step,lr_phase=lr_phase,
                    loss=sum(r['loss']*r['weight'] for r in results),
                    per_step_ev=np.sum([np.array(r['depth'])*r['weight'] for r in results],axis=0).tolist(),
                    p=np.sum([np.array(r['p'])*r['weight'] for r in results],axis=0).tolist(),
                    graphs=len(items),sample_indices=indices,sample_order_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                    edge_budget_accepted_cost=sum(costs),direct_batch=retries==0,
                    microbatch_sizes=[len(m) for m in micros],microbatch_costs=[sum(costs[i] for i in m) for m in micros],
                    cost_budget=controller.budget,oom_retries=retries,oom_total=controller.oom_count,
                    peak_allocated=peak,peak_reserved=reserved,device_free=free,device_total=total,
                    grad_norm=norm,counts=results[0]['counts'])
                history.write(json.dumps(row,allow_nan=False)+'\n')
                print('TRAIN',json.dumps(row),flush=True)
                del results,items,micros
                optimizer.zero_grad(set_to_none=True)
                if step==1 or step%args.checkpoint_every==0:save('step_%06d'%step)
            del accepted_batches,loader
            if sampler.cursor==len(sampler.order):sampler.next_epoch()
    save('joint_final')
    if stopping[0]:
        atomic_json(root/'paused.json',dict(step=step,reason='SIGUSR1 after committed optimizer update'))
        return
    validate('joint_final')
    if args.K>1 and args.gate_calibration>0:
        for name,p in model.named_parameters():p.requires_grad_('.stack_exit.' in name)
        model.eval();gate_opt=torch.optim.AdamW(groups['gate'],lr=1e-3,weight_decay=0,foreach=False)
        cal_losses=[]
        # Single graph calibration bounds memory; its labels remain TRAIN-only.
        for i,index in enumerate(cal_indices[:args.gate_calibration]):
            batch=next(iter(DataLoader(dataset=[datasets['train'][index]],batch_size=1,exclude_keys=exclude)))
            ref=AtomicData.to_AtomicDataDict(batch.cuda());gate_opt.zero_grad(set_to_none=True)
            out=model(dict(ref));ls=losses_for(out,ref,idp).detach()
            target=torch.sigmoid(5000*((ls[:,:-1]-ls[:,1:]).clamp_min(0)-.0002))
            logits=out['_stack_logits'][:,1:-1]
            if not logits.shape[1]:break
            offsets=logits.new_tensor([math.log(args.K-t-1) for t in range(1,args.K-1)])
            loss=torch.nn.functional.binary_cross_entropy_with_logits(-(logits-offsets),target[:,:-1])
            loss.backward();gate_opt.step();cal_losses.append(float(loss.detach()))
        atomic_json(root/'calibration.json',dict(losses=cal_losses,unit='one_training_graph_per_update',indices=cal_indices[:len(cal_losses)]))
        save('calibrated_final','gate_calibrated');validate('calibrated_final')
    atomic_json(root/'complete.json',dict(steps=step,parent_step=parent_step,completed_at=time.time(),deadline=deadline))
    print('DYNAMIC_ANNEAL_COMPLETE',step,flush=True)


if __name__=='__main__':main()
