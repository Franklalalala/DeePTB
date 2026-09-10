"""Bounded single-training-structure whole-stack fitting diagnostic.

Reports full-path bands independently of the matrix-only training objective.
This is an optimization canary, not a generalization experiment.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.loopscf.evaluate import (
    torch, A, AtomicData, DataLoader, build_dataset, build_model, OrbitalMapper,
    normalize, clone, plain, file_sha256, path_bands, rotation_check,
)
from dptb.nnops.loss import _nrme_mask, _erme_mask
from dptb.nnops.loopscf.spectral import _fw10_one_graph
from dptb.nnops.loopscf.stack import install_stack_loop, graph_hamiltonian_losses, adaptive_objective
from examples.loopscf.anneal_stack import atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--lr', type=float, default=3e-5)
    args = ap.parse_args()
    if args.K < 1 or args.steps < 1 or args.lr <= 0:
        ap.error('K, steps and lr must be positive')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        raise RuntimeError('Use a fresh run directory')
    torch.set_num_threads(6)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = normalize(json.loads(Path(args.input).read_text()))
    common = dict(cfg['common_options'], device='cuda:0')
    data = cfg['data_options']
    ds = build_dataset(**data['train'], r_max=data.get('r_max'),
                       er_max=data.get('er_max'), oer_max=data.get('oer_max'), **common)
    item = ds[args.index]
    kpts = plain(item[A.KPOINT_KEY]).cuda()
    eig_ref = plain(item[A.ENERGY_EIGENVALUE_KEY]).cpu()
    batch = next(iter(DataLoader(dataset=[item], batch_size=1, num_workers=0,
                     exclude_keys=[A.KPOINT_KEY, A.ENERGY_EIGENVALUE_KEY])))
    ref = AtomicData.to_AtomicDataDict(batch.cuda())
    ne = float(ref['nelec'].reshape(-1)[0])
    model = build_model(checkpoint=args.base_model, model_options=cfg['model_options'],
                        common_options=common).cuda()
    idp = OrbitalMapper(common['basis'], method='e3tb', device='cuda:0')
    torch.manual_seed(42)
    install_stack_loop(model, idp, K=args.K)
    import hashlib
    def tensor_hash(t):
        return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    groups = {'body': [], 'bridge': [], 'gate': []}
    hashes = {k: hashlib.sha256() for k in groups}
    for name, p in model.named_parameters():
        if p.requires_grad:
            group = 'bridge' if '.stack_bridge.' in name else 'gate' if '.stack_exit.' in name else 'body'
            groups[group].append(p)
            hashes[group].update(name.encode())
            hashes[group].update(bytes.fromhex(tensor_hash(p)))
    opt = torch.optim.AdamW([{'params': ps, 'lr': args.lr * (10 if g == 'bridge' else 1),
                              'multiplier': 10 if g == 'bridge' else 1}
                             for g, ps in groups.items()], weight_decay=.01)
    atomic_json(root / 'protocol.json', dict(vars(args), seed=42, gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__, dataset_root=data['train']['root'], n_atoms=len(ref[A.ATOM_TYPE_KEY]),
        nelec=ne, n_kpoints=len(kpts), base_sha256=file_sha256(args.base_model),
        config_sha256=file_sha256(args.input), source_sha256=file_sha256(__file__),
        initial_trainable_sha256={g: h.hexdigest() for g,h in hashes.items()},
        graph_tensor_sha256={k:tensor_hash(v) for k,v in ref.items() if torch.is_tensor(v)},
        objective='0.8*expected_graph_H_loss+0.2*mean_depth_H_loss-0.0005*exit_entropy',
        schedule='10-step linear warmup then cosine to 3 percent; bridge lr x10',
        target='AO residual dH; H0 added exactly once in bands; corrected H0 entry',
        scope='single fixed training record; no generalization claim'))
    at = ref[A.ATOM_TYPE_KEY].reshape(-1)
    masks = [_nrme_mask(idp, at, result_device=at.device),
             _erme_mask(idp, ref[A.EDGE_TYPE_KEY].flatten(), result_device=at.device)]

    @torch.no_grad()
    def evaluate(step, rotation=False):
        model.eval()
        out = model(clone(ref))
        outputs = {'label': (ref[A.NODE_FEATURES_KEY], ref[A.EDGE_FEATURES_KEY])}
        outputs.update({'fit_K%d'%k: pair for k,pair in enumerate(out['_loop_preds'],1)})
        spectra, overlap = path_bands(ref, outputs, idp, kpts, 8)
        closure = float(_fw10_one_graph(spectra['label'], eig_ref, ne, 10.)[0])
        if closure >= 1e-3:
            raise ValueError('label+H0 spectral closure failed: %g'%closure)
        row = {'step': step, 'label_closure_fw10_ev': closure, 'overlap':overlap,
               'exit_probabilities':out['_exit_probabilities'].tolist(), 'counts':out['_stack_counts'], 'variants':{}}
        for name,pair in list(outputs.items())[1:]:
            errs = [(p-ref[f]).abs()[m] for p,f,m in zip(pair,[A.NODE_FEATURES_KEY,A.EDGE_FEATURES_KEY],masks)]
            row['variants'][name] = {'onsite_mae_ev':float(errs[0].mean()),
                'hopping_mae_ev':float(errs[1].mean()), 'packed_mae_ev':float(torch.cat(errs).mean()),
                'matrix_loss_ev':float(graph_hamiltonian_losses(*pair,ref,idp)[0]),
                'fw10_ev':float(_fw10_one_graph(spectra[name],eig_ref,ne,10.)[0])}
        if rotation:
            row['rotation'] = rotation_check(model,ref,outputs,idp,list(range(1,args.K+1)),'fit')
        atomic_json(root / ('evaluation_%04d.json'%step), row)
        print('EVALUATION',json.dumps(row),flush=True)
        return row

    initial = evaluate(0, rotation=True)
    start = time.monotonic()
    for step in range(1,args.steps+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        scale = min(1.,step/10) * (.03 + .97*.5*(1+math.cos(math.pi*max(0,step-10)/max(1,args.steps-10))))
        for g in opt.param_groups:
            g['lr'] = args.lr * g['multiplier'] * scale
        out = model(clone(ref))
        losses = torch.stack([graph_hamiltonian_losses(n,e,ref,idp) for n,e in out['_loop_preds']],-1)
        expected,entropy = adaptive_objective(losses,out['_exit_probabilities'],0.)
        loss = .8*expected + .2*losses.mean() - .0005*entropy
        if not torch.isfinite(loss):
            raise ValueError('nonfinite objective')
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        opt.step()
        row = {'step':step,'loss_ev':float(loss.detach()),'depth_matrix_loss_ev':losses.detach().tolist(),
               'exit_probabilities':out['_exit_probabilities'].detach().tolist(),'gradient_norm':norm,
               'lr':args.lr*scale,'elapsed_seconds':time.monotonic()-start,
               'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        with (root/'history.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        if step == 1 or step % 10 == 0:
            print('UPDATE',json.dumps(row),flush=True)
        if step % 50 == 0 or step == args.steps:
            final = evaluate(step, rotation=step==args.steps)
    checkpoint = {'model_state_dict':model.state_dict(),'stack_protocol':{'K':args.K,'strategy':'bptt'},
                  'step':args.steps,'base_model':args.base_model}
    torch.save(checkpoint,root/'final.tmp')
    (root/'final.tmp').replace(root/'final.pth')
    atomic_json(root/'complete.json',{'initial':initial,'final':final,'elapsed_seconds':time.monotonic()-start,
                                    'checkpoint_sha256':file_sha256(root/'final.pth')})
    print('SINGLE_STRUCTURE_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
