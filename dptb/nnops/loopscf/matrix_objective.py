"""Graph allocation of native L1/RMSE loss for adaptive matrix recurrence.

Each graph receives its exact additive contribution to the native batch loss:
L1_g/N and SSE_g/(N*RMSE_batch). Summing graphs recovers the native loss,
including its batch RMSE (not a mean of per-graph RMSE). Uniform exits thus
recover the mean native round loss, and K=1 recovers the base objective.
"""
import torch
from dptb.data import AtomicDataDict as A
from dptb.nnops.loss import _nrme_mask, _erme_mask
from dptb.nnops.layout import normalize_idp_mask_layout, project_uureal_to_like
from .stack import adaptive_objective


def native_graph_contributions(data, ref, idp, *, intrinsic=False):
    batch = data[A.BATCH_KEY].flatten()
    ng = int(batch.max()) + 1
    result = data[A.NODE_FEATURES_KEY].new_zeros(ng)
    for key, types, getter, physical, groups in (
        (A.NODE_FEATURES_KEY,A.ATOM_TYPE_KEY,_nrme_mask,"expert_node_mask",batch),
        (A.EDGE_FEATURES_KEY,A.EDGE_TYPE_KEY,_erme_mask,"expert_edge_mask",batch[data[A.EDGE_INDEX_KEY][0]]),
    ):
        pred,_ = project_uureal_to_like(idp,data[key],ref[key])
        mask = getter(idp,data[types].flatten(),result_device=pred.device)
        if physical in data:
            mask = mask & data[physical].reshape(-1,1)
        mask = normalize_idp_mask_layout(idp,mask,ref[key],label=key)
        diff = (pred-ref[key])*mask
        raw_count = mask.sum().to(pred.dtype)
        count = raw_count.clamp_min(1)
        count_g = result.new_zeros(ng).index_add(0,groups,mask.sum(-1).to(pred.dtype))
        abs_g = result.new_zeros(ng).index_add(0,groups,diff.abs().sum(-1))
        sq_g = result.new_zeros(ng).index_add(0,groups,diff.square().sum(-1))
        if intrinsic:
            valid=(count_g>0.5).to(pred.dtype)
            result=result+0.25*(abs_g/count_g.clamp_min(1)+(sq_g/count_g.clamp_min(1)+1e-12).sqrt())*valid
            continue
        mse = sq_g.sum()/count
        # Native HamilLossAbs adds 1e-12 even at perfect prediction. Allocate
        # this floor by each graph's active count to retain K1 gradients too.
        rmse = (mse+1e-12).sqrt()
        rmse_g = (sq_g+1e-12*count_g)/(count*rmse)
        rmse_g = rmse_g*(raw_count>0.5).to(pred.dtype)
        result = result + 0.25*(abs_g/count+rmse_g)
    return result


def attach_adaptive_matrix_loss(criterion, beta_relative=0.05, adaptive_weight=0.8):
    """Native parameter objective, intrinsic graph signal for the exit gate.

    A zero-valued surrogate routes gate gradients through each graph's own
    masked mean error. This avoids other graphs' RMSE denominator changing
    the preferred exit of an improving graph. Entropy has a relative scale.
    """
    if getattr(criterion,"onsite_boost",False) or getattr(criterion,"element_average",False) or criterion.z_loss_coef:
        raise ValueError("adaptive matrix loss requires unboosted native L1/RMSE")
    if any(getattr(criterion,k,0) for k in ('router_z_loss_coef','router_aux_loss_coef')):
        raise ValueError('router regularization is outside the matrix-only objective')
    original = criterion.forward
    def forward(data,ref):
        rounds=data.get("_loop_preds")
        if rounds is None:
            return original(data,ref)
        losses=[];contributions=[];intrinsic=[]
        for node,edge in rounds:
            pred=dict(data);pred[A.NODE_FEATURES_KEY]=node;pred[A.EDGE_FEATURES_KEY]=edge
            losses.append(original(pred,ref))
            contributions.append(native_graph_contributions(pred,ref,criterion.idp))
            with torch.no_grad():intrinsic.append(native_graph_contributions(pred,ref,criterion.idp,intrinsic=True))
        loss=torch.stack(contributions,-1)
        gate_loss=torch.stack(intrinsic,-1)
        probabilities=data['_exit_probabilities']
        beta=beta_relative*gate_loss.mean().detach().clamp_min(1e-12)
        gate_task,entropy=adaptive_objective(gate_loss,probabilities,beta)
        # Preserve the reported native weighted value, using the intrinsic
        # signal only for the gate gradient. K1 value and gradients are exact.
        native_task=(probabilities.detach()*loss).sum()
        task=native_task+(gate_task-gate_task.detach())-beta*entropy.mean().detach()
        criterion.depth_diagnostics={"entropy":entropy.detach(),"probabilities":data['_exit_probabilities'].detach(),
                                     "round_native_losses":torch.stack(losses).detach(),"graph_contributions":loss.detach(),
                                     "gate_intrinsic_losses":gate_loss,"beta_effective":beta,"beta_relative":beta_relative}
        return adaptive_weight*task+(1-adaptive_weight)*torch.stack(losses).mean()
    criterion.forward=forward
    return criterion
