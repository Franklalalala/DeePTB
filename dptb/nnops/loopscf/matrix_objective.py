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


def native_graph_contributions(data, ref, idp):
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
        count = mask.sum().to(pred.dtype).clamp_min(1)
        abs_g = result.new_zeros(ng).index_add(0,groups,diff.abs().sum(-1))
        sq_g = result.new_zeros(ng).index_add(0,groups,diff.square().sum(-1))
        mse = sq_g.sum()/count
        rmse = mse.clamp_min(1e-24).sqrt()
        rmse_g = torch.where(mse>0,sq_g/(count*rmse),torch.zeros_like(sq_g))
        result = result + 0.25*(abs_g/count+rmse_g)
    return result


def attach_adaptive_matrix_loss(criterion, beta=0.0005, adaptive_weight=0.8):
    """Retain native masks, metric accumulators and exact K=1 reduction."""
    if getattr(criterion,"onsite_boost",False) or getattr(criterion,"element_average",False) or criterion.z_loss_coef:
        raise ValueError("adaptive matrix loss requires unboosted native L1/RMSE")
    original = criterion.forward
    def forward(data,ref):
        rounds=data.get("_loop_preds")
        if rounds is None:
            return original(data,ref)
        losses=[];contributions=[]
        for node,edge in rounds:
            pred=dict(data);pred[A.NODE_FEATURES_KEY]=node;pred[A.EDGE_FEATURES_KEY]=edge
            losses.append(original(pred,ref))
            contributions.append(native_graph_contributions(pred,ref,criterion.idp))
        loss=torch.stack(contributions,-1)
        # adaptive_objective averages graphs; restore native sum scaling.
        task,entropy=adaptive_objective(loss*loss.shape[0],data['_exit_probabilities'],beta)
        criterion.depth_diagnostics={"entropy":entropy.detach(),"probabilities":data['_exit_probabilities'].detach(),
                                     "round_native_losses":torch.stack(losses).detach(),"graph_contributions":loss.detach()}
        return adaptive_weight*task+(1-adaptive_weight)*torch.stack(losses).mean()
    criterion.forward=forward
    return criterion
