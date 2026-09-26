"""Single-GPU matrix-depth continuation with evidence and bounded final evaluation.

Uses the externally pinned India exact-state helper for sampler/RNG snapshots.
The helper and runtime remain read-only; every result is under --run.
"""
import argparse,copy,hashlib,heapq,json,math,os,pathlib,signal,socket,subprocess,sys,time


def configure_task_ipc():
    """Long task-only TMPDIR cannot hold multiprocessing AF_UNIX paths.

    Linux abstract sockets retain multiprocessing authentication and create
    no filesystem object outside (or inside) the task directory.
    """
    if sys.platform!='linux':return
    import multiprocessing.connection as connection
    import uuid
    original=connection.arbitrary_address
    def address(family):
        if family=='AF_UNIX':return '\0loopdepth0927-'+str(os.getpid())+'-'+uuid.uuid4().hex
        return original(family)
    connection.arbitrary_address=address


configure_task_ipc()


def main():
    p=argparse.ArgumentParser()
    for name in ('config','base','run','state-helper'):
        p.add_argument('--'+name,required=True,type=pathlib.Path)
    p.add_argument('--mode',choices=['dense','stack','core','unshared','latent'],required=True)
    p.add_argument('--head',choices=['onsite','hopping'],required=True)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--eval-limit',type=int,default=3000)
    p.add_argument('--resume',type=pathlib.Path)
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--start-file',type=pathlib.Path)
    a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.state_helper))
    from exact_state import atomic_json,atomic_torch,append,loader_contract,epoch_iterator,snapshot,restore,digest
    import torch
    import dptb
    from dptb.data import AtomicDataDict as A
    from dptb.entrypoints.multi_train import multi_train
    from dptb.nnops.multi_trainer import MultiTrainer
    from dptb.plugins.saver import Saver
    from dptb.plugins.monitor import Validationer
    from dptb.utils.tools import get_optimizer,get_lr_scheduler
    from dptb.nnops.loopscf.matrix_depth import install_matrix_depth,ReplayDepth,clone_data,matrix_predict_until_exit
    from dptb.nnops.loopscf.stack import exit_distribution
    from dptb.nnops.loopscf.matrix_objective import attach_adaptive_matrix_loss
    assert torch.cuda.device_count()==1
    assert os.environ.get('SLURM_JOB_ID')=='81373'
    assert socket.gethostname() in ('cbgpu0086','cbgpu0235','cbgpu0236')
    config=json.loads(a.config.read_text())
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    parent_step=int(base['iteration'])
    identity={'code':subprocess.check_output(['git','-C',str(pathlib.Path(dptb.__file__).parents[1]),'rev-parse','HEAD'],text=True).strip(),
              'base_sha256':hashlib.sha256(a.base.read_bytes()).hexdigest(),
              'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
              'mode':a.mode,'head':a.head,'parent_step':parent_step,
              'state_helper_sha256':hashlib.sha256((a.state_helper/'exact_state.py').read_bytes()).hexdigest()}
    atomic_json(a.run/'START.json',dict(identity,pid=os.getpid(),host=socket.gethostname(),
                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),time=time.time(),deadline=a.deadline,eval_limit=a.eval_limit))
    stopped=[False]
    def stop(*_):stopped[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    Saver.iteration=lambda *args,**kw:None
    Saver.epoch=lambda *args,**kw:None
    Validationer.iteration=lambda *args,**kw:None
    Validationer.epoch=lambda *args,**kw:None

    def maps_receipt(iterator):
        pids=[os.getpid()]+[w.pid for w in getattr(iterator,'_workers',[]) if w.pid]
        rows=[]
        for pid in pids:
            f=pathlib.Path('/proc')/str(pid)/'maps'
            lines=f.read_text().splitlines()
            shared=[line for line in lines if 'data.mdb' in line and '/mkliu/data/' in line]
            rows.append({'pid':pid,'shared_data_mdb':shared})
        assert all(not r['shared_data_mdb'] for r in rows),rows
        assert len(pids)>1,'Must inspect live DataLoader workers'
        return rows

    def evaluate(t,label,limit):
        t.model.eval();begin=time.monotonic();rows=0
        # Native validation shuffles in single-process mode. Fix only its
        # traversal order so every saved row is the same dataset index.
        seq=torch.utils.data.SequentialSampler(t.validation_datasets)
        object.__setattr__(t.validation_loader,'sampler',seq)
        t.validation_loader.batch_sampler.sampler=seq
        t.validation_loader_generator.manual_seed(t.validation_loader_seed)
        if a.mode!='dense':t.model._matrix_depth_K=6
        lf=t.validation_lossfunc
        sums=[dict(absolute=0.,square=0.,count=0.) for _ in range(6)]
        path=a.run/(label+'.jsonl')
        if path.exists():raise RuntimeError('Evaluation output already exists: '+str(path))
        with torch.no_grad():
            for batch in t.validation_loader:
                if rows>=limit:break
                data,info=t._prepare_batch_bundle(batch,with_lengths=True)
                pred=t.model(clone_data(data));pred.update(info)
                edge_mask,node_mask=t._prepare_expert_masks(data,t.distance_ranges[0],0)
                pred['expert_edge_mask']=edge_mask;pred['expert_node_mask']=node_mask
                ref=dict(data);ref.update(info)
                predictions=pred.get('_loop_preds',[(pred[A.NODE_FEATURES_KEY],pred[A.EDGE_FEATURES_KEY])]*6)
                curve=[]
                for k,(node,edge) in enumerate(predictions):
                    one=dict(pred);one[A.NODE_FEATURES_KEY]=node;one[A.EDGE_FEATURES_KEY]=edge
                    loss=lf(one,ref)
                    stats={name:float(getattr(lf,'last_'+a.head+'_'+key)) for name,key in
                           [('absolute','l1_sum'),('square','mse_sum'),('count','count')]}
                    assert stats['count']>0
                    curve.append(dict(k=k+1,loss=float(loss),mae=stats['absolute']/stats['count'],**stats))
                    for key in stats:sums[k][key]+=stats[key]
                probs=pred.get('_exit_probabilities')
                probs3=exit_distribution(pred['_stack_logits'][...,:3]) if probs is not None else None
                early=None
                if a.mode!='dense' and rows<8:
                    begin_exit=time.monotonic()
                    deployed=matrix_predict_until_exit(t.model,clone_data(data),3,0.5)
                    torch.cuda.synchronize()
                    selected=deployed['_exit_step']-1
                    for key,expected in zip((A.NODE_FEATURES_KEY,A.EDGE_FEATURES_KEY),predictions[selected]):
                        torch.testing.assert_close(deployed[key],expected,atol=1e-6,rtol=1e-5)
                    early={'step':selected+1,'cdf':deployed['_exit_cdf'],'seconds':time.monotonic()-begin_exit}
                append(path,{'index':rows,'curve':curve,'probabilities':probs.cpu().tolist() if probs is not None else None,
                             'probabilities_k3':probs3.cpu().tolist() if probs3 is not None else None,'actual_early_exit':early,
                             'atom_count':int(data[A.ATOM_TYPE_KEY].numel()),'edge_count':int(data[A.EDGE_INDEX_KEY].shape[1])})
                rows+=1
        torch.cuda.synchronize()
        result={'records':rows,'seconds':time.monotonic()-begin,'step':int(t.iter-1),
                'metrics':[dict(k=i+1,mae=v['absolute']/v['count'],rmse=math.sqrt(v['square']/v['count']),**v) for i,v in enumerate(sums)]}
        atomic_json(a.run/(label+'.json'),result)
        return result

    def run(t,epochs=1):
        assert len(t.optimizers)==1 and not t.distributed_expert
        assert len(t.optimizers[0].param_groups)==1, 'Update witness requires the pinned single-group optimizer'
        sampler=loader_contract(t)
        atomic_json(a.run/'DATA.json',{'train_records':len(t.train_datasets),'test_records':len(t.validation_datasets),
                                     'dynamic_batch':t.train_loader.dynamic_batch_options})
        assert len(t.train_datasets)==29242 and len(t.validation_datasets)==3000
        # Cache measured immutable per-record costs, never a hand-set budget.
        # Native 128-batch calibration has already run above and stays intact.
        root=a.run
        while root.name!='loopdepth0927' and root!=root.parent:root=root.parent
        assert root.name=='loopdepth0927'
        cost_identity={'data':config['data_options']['train'],'basis':config['common_options']['basis'],
                       'dataset_gate':hashlib.sha256((root/'outputs/DATASET_GATE.json').read_bytes()).hexdigest(),
                       'cost_signature':sampler._cost_signature(),'records':len(t.train_datasets)}
        cost_file=root/'cache'/('main_costs_'+digest(cost_identity)+'.json')
        if cost_file.exists():
            cached=json.loads(cost_file.read_text());assert cached['identity_hash']==digest(cost_identity)
            costs={int(k):v for k,v in cached['costs'].items()}
            assert set(costs)==set(range(len(t.train_datasets)))
            from dptb.data.dataloader import _metadata_cost_parts
            for idx in [0,1,127,128,731,12345,29241]:
                assert _metadata_cost_parts(t.train_datasets,idx,sampler.cost_estimator)[0]==costs[idx]
            sampler._cost_cache=costs
            atomic_json(a.run/'COST_CACHE.json',{'loaded':str(cost_file),'identity_hash':digest(cost_identity),'sample_verified':7})
        # Restore original moments before adding parameters. Native continuation
        # of these finished WSD checkpoints stays at min_lr; no silent restart.
        t.optimizers[0].load_state_dict(base['optimizers_state_dict'][0])
        t.lr_schedulers[0].load_state_dict(base['lr_schedulers_state_dict'][0])
        old_optimizer=t.optimizers[0]
        t.iter=parent_step+1;t.ep=1
        t.train_options['max_steps']=parent_step+a.steps
        depth_rng=ReplayDepth(3,config['common_options']['seed'])
        iterator,plan,cursor,epoch_rng=epoch_iterator(t,sampler)
        if len(sampler._cost_cache)==len(t.train_datasets):
            atomic_json(cost_file,{'identity_hash':digest(cost_identity),
                                  'costs':{int(k):float(v) for k,v in sampler._cost_cache.items()}})
            if not (a.run/'COST_CACHE.json').exists():
                atomic_json(a.run/'COST_CACHE.json',{'created':str(cost_file),
                            'identity_hash':digest(cost_identity),'records':len(sampler._cost_cache)})
        first=next(iterator)
        data,info=t._prepare_batch_bundle(first,with_lengths=True)
        t.model.eval()
        with torch.no_grad():expected=t.model(clone_data(data))
        if a.mode!='dense':
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(config['common_options']['seed'])
                install_matrix_depth(t.model,a.mode,maximum=6)
            t.model._matrix_depth_K=1
        with torch.no_grad():actual=t.model(clone_data(data))
        errors={}
        for key in (A.NODE_FEATURES_KEY,A.EDGE_FEATURES_KEY):
            errors[key]=float((expected[key]-actual[key]).abs().max())
            torch.testing.assert_close(actual[key],expected[key],atol=1e-6,rtol=1e-5)
        atomic_json(a.run/'K1_GATE.json',dict(passed=True,max_abs=errors,structures=len(plan[0]),parent_step=parent_step))
        del expected,actual,data,info
        if a.mode!='dense':
            optimizer=get_optimizer(model_param=t._expert_optimizer_parameters(0),**t._build_optimizer_cfg_for_expert(0))
            assert len(optimizer.param_groups)==len(old_optimizer.param_groups)
            for new,old in zip(optimizer.param_groups,old_optimizer.param_groups):
                for key,value in old.items():
                    if key!='params':new[key]=copy.deepcopy(value)
                for param in new['params']:
                    if param in old_optimizer.state:optimizer.state[param]=old_optimizer.state[param]
            scheduler=get_lr_scheduler(optimizer=optimizer,**config['train_options']['lr_scheduler'])
            scheduler.load_state_dict(t.lr_schedulers[0].state_dict())
            for new,old in zip(optimizer.param_groups,old_optimizer.param_groups):new['lr']=old['lr']
            t.optimizers=[optimizer];t.lr_schedulers=[scheduler]
            attach_adaptive_matrix_loss(t.train_lossfunc)
        del old_optimizer
        saved=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
        if saved:
            assert saved['depth_mode']==a.mode
            restore(t,saved,identity);depth_rng.load_state_dict(saved['depth_rng'])
            del iterator
            if a.evaluate_only:
                # Validate the immutable plan without reading its consumed prefix.
                # Evaluation has no dependency on training-worker RNG or cursor.
                t._set_expert_dp_sampler_epoch(t.ep)
                plan=list(sampler);assert plan==saved['batch_plan'], 'Sampler/data order changed'
                cursor=saved['consumed_batches'];assert 0<=cursor<=len(plan)
                epoch_rng=saved['epoch_rng'];iterator=None
                atomic_json(a.run/'EVAL_RESTORE.json',{'plan_verified':True,'prefix_batches_read':0,'saved_cursor':cursor})
            else:
                iterator,plan,cursor,epoch_rng=epoch_iterator(t,sampler,saved)
            first=None
        t.rebase_plugin_cadence()
        for q in t.plugin_queues.values():heapq.heapify(q)
        last_iteration_state={}
        original_call_plugins=t.call_plugins
        def capture_plugins(*args,**kwargs):
            if kwargs.get('queue_name')=='iteration':last_iteration_state.update(kwargs)
            return original_call_plugins(*args,**kwargs)
        t.call_plugins=capture_plugins
        atomic_json(a.run/'OPTIMIZER.json',{'lr':[g['lr'] for g in t.optimizers[0].param_groups],
                  'scheduler_last_epoch':t.lr_schedulers[0].last_epoch,'trainable':sum(p.numel() for p in t.model.parameters() if p.requires_grad),
                  'parameters':sum(p.numel() for p in t.model.parameters()),'restored_parameter_states':len(t.optimizers[0].state)})
        if a.start_file and not a.evaluate_only:
            atomic_json(a.run/'READY.json',dict(identity,pid=os.getpid(),time=time.time()))
            while not a.start_file.exists():
                if stopped[0] or time.time()>=a.deadline:raise RuntimeError('formal start barrier expired')
                time.sleep(1)
            release=json.loads(a.start_file.read_text());assert not release.get('abort'),release
            atomic_json(a.run/'TRAIN_START.json',dict(time=time.time(),release=release))
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();last_save=start;commits=0;fetch_start=time.monotonic()
        while not a.evaluate_only:
            if first is not None:batch=first;first=None
            else:
                try:batch=next(iterator)
                except StopIteration:
                    del iterator;t.call_plugins(queue_name='epoch',time=t.ep);t.update();t.ep+=1
                    iterator,plan,cursor,epoch_rng=epoch_iterator(t,sampler)
                    batch=next(iterator)
            fetch_seconds=time.monotonic()-fetch_start
            batch_ids=plan[cursor];cursor+=1
            sampled=depth_rng.sample()
            depth=1 if a.mode=='dense' else sampled
            if a.mode!='dense':t.model._matrix_depth_K=depth
            before=t.iter;begin=time.monotonic()
            witness=commits in (0,19,99)
            previous={n:p.detach().clone() for n,p in t.model.named_parameters() if p.requires_grad} if witness else None
            previous_lr=t.optimizers[0].param_groups[0]['lr']
            loss=t.iteration(batch);torch.cuda.synchronize()
            if t.iter==before:
                append(a.run/'SKIPPED.jsonl',{'step':before,'cursor':cursor,'K':depth})
                fetch_start=time.monotonic();continue
            assert t.iter==before+1 and math.isfinite(float(loss))
            commits+=1
            row={'step':before,'relative_step':before-parent_step,'epoch':t.ep,'K':depth,'indices':batch_ids,
                 'loss':float(loss),'fetch_seconds':fetch_seconds,'iteration_seconds':time.monotonic()-begin,
                 'lr':[g['lr'] for g in t.optimizers[0].param_groups],
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                 'time':time.time(),'pid':os.getpid(),'host':socket.gethostname()}
            row['gradient_norm_before_clip']=float(last_iteration_state['total_grad_norm'])
            row['gradient_clip_limit']=float(t.clip_grad_norm)
            if witness:
                deltas=[(p.detach()-previous[n]).square().sum() for n,p in t.model.named_parameters() if n in previous]
                row['parameter_delta_l2']=float(torch.stack(deltas).sum().sqrt());assert row['parameter_delta_l2']>0
                assert all(torch.isfinite(p).all() for p in t.model.parameters())
                row['maps']=maps_receipt(iterator)
                row['nonzero_gradient_tensors']=sum(int(p.grad is not None and bool(p.grad.abs().sum()>0)) for p in t.model.parameters())
                assert row['nonzero_gradient_tensors']>0
                group=t.optimizers[0].param_groups[0]
                assert group.get('expert_lr_mult',1)==1 and group.get('expert_weight_decay_mult',1)==1
                wd=group['weight_decay']
                nondecay=[(p.detach()-previous[n]*(1-previous_lr*wd if p.grad is not None else 1)).square().sum()
                          for n,p in t.model.named_parameters() if n in previous]
                row['nondecay_parameter_delta_l2']=float(torch.stack(nondecay).sum().sqrt())
                assert row['nondecay_parameter_delta_l2']>0
                row['optimizer_diagnostics']=t.optimizers[0].get_diagnostics()
                gate_deltas=[(p.detach()-previous[n]).square().sum() for n,p in t.model.named_parameters() if n in previous and '.depth_gate.' in n]
                row['gate_parameter_delta_l2']=float(torch.stack(gate_deltas).sum().sqrt()) if gate_deltas else 0.
                del nondecay,gate_deltas
                del previous,deltas
            diag=getattr(t.train_lossfunc,'depth_diagnostics',None)
            if diag:
                row['entropy_mean']=float(diag['entropy'].mean());row['exit_mean']=diag['probabilities'].mean(0).tolist()
                row['round_native_losses']=diag['round_native_losses'].tolist()
                row['beta_effective']=float(diag['beta_effective']);row['beta_relative']=diag['beta_relative']
            append(a.run/'STEPS.jsonl',row)
            if commits%10==0 or witness:atomic_json(a.run/'HEARTBEAT.json',row)
            final=stopped[0] or time.time()>=a.deadline or before>=parent_step+a.steps
            if final or time.monotonic()-last_save>=1800 or (before-parent_step)%1000==0:
                state=snapshot(t,identity,plan,cursor,epoch_rng);state['depth_rng']=depth_rng.state_dict();state['depth_mode']=a.mode
                path=a.run/'checkpoints'/('step-%08d.pth'%(before-parent_step))
                atomic_torch(path,state);atomic_json(a.run/'LATEST.json',{'step':before,'relative_step':before-parent_step,'checkpoint':str(path)})
                last_save=time.monotonic()
            if final:break
            fetch_start=time.monotonic()
        del iterator
        result=evaluate(t,'FINAL_K',a.eval_limit)
        atomic_json(a.run/'RESULT.json',{'status':'evaluated','committed_step':t.iter-1,'relative_step':t.iter-1-parent_step,
                                      'train_seconds':time.monotonic()-start-result['seconds'],'evaluation':result,'finish_time':time.time()})
    MultiTrainer.run=run
    multi_train(INPUT=str(a.config),init_model=str(a.base),restart=None,output=str(a.run/'native'),log_level=20,log_path=str(a.run/'train.log'))


if __name__=='__main__':main()
