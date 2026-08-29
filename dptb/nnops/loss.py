import torch.nn as nn
import torch
from torch.nn.functional import mse_loss
from dptb.utils.register import Register
from dptb.nn.energy import Eigenvalues
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.nnops.layout import normalize_idp_mask_layout, project_uureal_to_like
from typing import Any, Union, Dict
from dptb.data import AtomicDataDict, AtomicData
from dptb.data.transforms import OrbitalMapper
from e3nn.o3 import Irreps
from torch_scatter import scatter_mean
from dptb.utils.torch_geometric import Batch
import matplotlib.pyplot as plt
from dptb.utils.constants import anglrMId
from dptb.nn.dftbsk import DFTBSK
import re
from dptb.nn.hr2hk import HR2HK, HR2HK_Gamma_Only
from dptb.utils.soc_target import resolve_nextham_uureal_mask
# from pyscf import gto, dft

"""this is the register class for descriptors

all descriptors inplemendeted should be a instance of nn.Module class, and provide a forward function that
takes AtomicData class as input, and give AtomicData class as output.

"""
class Loss:
    _register = Register()

    def register(target):
        return Loss._register.register(target)
    
    def __new__(cls, method: str, **kwargs):
        if method in Loss._register.keys():
            return Loss._register[method](**kwargs)
        else:
            raise Exception(f"Loss method: {method} is not registered!")


def _take_idp_tensor(
    tensor: torch.Tensor,
    index,
    result_device=None,
):
    if torch.is_tensor(index):
        tensor = tensor.to(device=index.device)
        out = tensor[index]
    else:
        out = tensor[index]

    if result_device is not None and torch.is_tensor(out):
        out = out.to(device=result_device)
    return out


def _nrme_mask(idp, atom_types, result_device=None):
    return _take_idp_tensor(idp.mask_to_nrme, atom_types, result_device=result_device)


def _erme_mask(idp, edge_types, result_device=None):
    return _take_idp_tensor(idp.mask_to_erme, edge_types, result_device=result_device)


def _basis_mask(idp, atom_type, result_device=None):
    return _take_idp_tensor(idp.mask_to_basis, atom_type, result_device=result_device)


def _masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    diff = pred - target
    count = mask.sum()
    return (diff.square() * mask).sum() / count.clamp_min(1.0)


def _l1_rmse_loss_from_sums(
    abs_sum: torch.Tensor,
    square_sum: torch.Tensor,
    count: torch.Tensor,
) -> torch.Tensor:
    valid = (count > 0.5).to(dtype=abs_sum.dtype)
    safe_count = count.clamp_min(1.0)
    l1_mean = abs_sum / safe_count
    mse_mean = square_sum / safe_count
    return 0.5 * (l1_mean + torch.sqrt(mse_mean + (1.0 - valid) + 1e-12)) * valid

@Loss.register("skints")
class DFTBskLoss(nn.Module):
    def __init__(
                self,
                basis: Dict[str, Union[str, list]]=None,
                skdata: str=None,
                overlap: bool = False,
                dtype: Union[str, torch.dtype] = torch.float32, 
                device: Union[str, torch.device] = torch.device("cpu"),
                **kwargs) -> None:
        
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        
        self.loss = nn.MSELoss()

        self.dftbsk = DFTBSK(basis=basis, skdata=skdata, overlap=overlap, dtype=dtype, device=device,transform=False)

        self.overlap = overlap
    
    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        total_loss = 0.
        ref_data = AtomicData.to_AtomicDataDict(ref_data)
        ref_data = self.dftbsk(ref_data)

        # onsite loss
        onsite_loss = mse_loss(data[AtomicDataDict.NODE_FEATURES_KEY], ref_data[AtomicDataDict.NODE_FEATURES_KEY])

        # hopping loss
        hopping_loss = mse_loss(data[AtomicDataDict.EDGE_FEATURES_KEY], ref_data[AtomicDataDict.EDGE_FEATURES_KEY])

        # overlap loss
        total_loss = onsite_loss + hopping_loss
        if self.overlap:
            total_loss = total_loss + mse_loss(data[AtomicDataDict.EDGE_OVERLAP_KEY], ref_data[AtomicDataDict.EDGE_OVERLAP_KEY])
        
        return total_loss

@Loss.register("eigvals")
class EigLoss(nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            diff_on: bool=False,
            eout_weight: float=0.01,
            diff_weight: float=0.01,
            diff_valence: dict=None,
            spin_deg: int = 2,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):
        super(EigLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.device = device
        self.diff_on = diff_on
        self.eout_weight = eout_weight
        self.diff_weight = diff_weight
        self.diff_valence = diff_valence  
        self.spin_deg = spin_deg  


        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        if not overlap:
            self.eigenvalue = Eigenvalues(
                idp=self.idp,
                h_edge_field = AtomicDataDict.EDGE_FEATURES_KEY,
                h_node_field = AtomicDataDict.NODE_FEATURES_KEY,
                h_out_field = AtomicDataDict.HAMILTONIAN_KEY,
                out_field = AtomicDataDict.ENERGY_EIGENVALUE_KEY,
                s_edge_field = None,
                s_node_field = None,
                s_out_field = None, 
                dtype=dtype, 
                device=device,
                )
        else:
            self.eigenvalue = Eigenvalues(
                idp=self.idp,
                h_edge_field = AtomicDataDict.EDGE_FEATURES_KEY,
                h_node_field = AtomicDataDict.NODE_FEATURES_KEY,
                h_out_field = AtomicDataDict.HAMILTONIAN_KEY,
                out_field = AtomicDataDict.ENERGY_EIGENVALUE_KEY,
                s_edge_field = AtomicDataDict.EDGE_OVERLAP_KEY,
                s_node_field = AtomicDataDict.NODE_OVERLAP_KEY,
                s_out_field = AtomicDataDict.OVERLAP_KEY, 
                dtype=dtype, 
                device=device,
                )

        self.overlap = overlap
    
    def forward(
            self, 
            data: AtomicDataDict, 
            ref_data: AtomicDataDict,
            ):
        
        total_loss = 0.

        data = Batch.from_dict(data)
        ref_data = Batch.from_dict(ref_data)

        datalist = data.to_data_list()
        ref_datalist = ref_data.to_data_list()
        for data, ref_data in zip(datalist, ref_datalist):
            data = self.eigenvalue(AtomicData.to_AtomicDataDict(data))
            ref_data = AtomicData.to_AtomicDataDict(ref_data)
            if ref_data.get(AtomicDataDict.ENERGY_EIGENVALUE_KEY) is None:
                ref_data = self.eigenvalue(ref_data)
            
            emin, emax = ref_data.get(AtomicDataDict.ENERGY_WINDOWS_KEY, (None, None))
            band_min, band_max = ref_data.get(AtomicDataDict.BAND_WINDOW_KEY, (0, None))
            eig_pred = data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0] # (n_kpt, n_band)
            eig_label = ref_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0] # (n_kpt, n_band_dft/n_band)

            if self.diff_valence is not None and isinstance(self.diff_valence, dict):
                nbands_exclude = sum([self.diff_valence[self.idp.type_to_chemical_symbol[int(ii)]] for ii in ref_data['atom_types']])
                assert nbands_exclude % self.spin_deg == 0
                nbands_exclude = nbands_exclude // self.spin_deg
            else:
                nbands_exclude = 0
            
            eig_label = eig_label[:,nbands_exclude:]

            norbs = eig_pred.shape[-1]
            nbanddft = eig_label.shape[-1]
            num_kp = eig_label.shape[-2]

            assert num_kp == eig_pred.shape[-2]
            up_nband = min(norbs, nbanddft)

            if band_max == None:
                band_max = up_nband
            else:
                assert band_max <= up_nband

            band_min = int(band_min)
            band_max = int(band_max)

            assert band_min < band_max
            assert len(eig_pred.shape) == 2 and len(eig_label.shape) == 2

            # 对齐eig_pred和eig_label
            eig_pred_cut = eig_pred[:,band_min:band_max]
            eig_label_cut = eig_label[:,band_min:band_max]


            num_kp, num_bands = eig_pred_cut.shape

            eig_pred_cut = eig_pred_cut - eig_pred_cut.reshape(-1).min()
            eig_label_cut = eig_label_cut - eig_label_cut.reshape(-1).min()

            
            if emax != None and emin != None:
                mask_in = eig_label_cut.lt(emax) * eig_label_cut.gt(emin)
                mask_out = eig_label_cut.gt(emax) + eig_label_cut.lt(emin)
            elif emax != None:
                mask_in = eig_label_cut.lt(emax)
                mask_out = eig_label_cut.gt(emax)
            elif emin != None:
                mask_in = eig_label_cut.gt(emin)
                mask_out = eig_label_cut.lt(emin)
            else:
                mask_in = None
                mask_out = None

            if mask_in is not None:
                loss = _masked_mse_loss(eig_pred_cut, eig_label_cut, mask_in)
                loss = loss + self.eout_weight * _masked_mse_loss(eig_pred_cut, eig_label_cut, mask_out)
            else:
                loss = mse_loss(eig_pred_cut, eig_label_cut)

            if self.diff_on:
                assert num_kp >= 1
                # randon choose nk_diff kps' eigenvalues to gen Delta eig.
                # nk_diff = max(nkps//4,1)     
                nk_diff = num_kp
                k_diff_i = torch.randint(0, num_kp, (nk_diff,), device=self.device)
                k_diff_j = torch.randint(0, num_kp, (nk_diff,), device=self.device)
                while (k_diff_i==k_diff_j).all():
                    k_diff_j = torch.randint(0, num_kp, (nk_diff,), device=self.device)
                if mask_in is not None:
                    eig_diff_lbl = eig_label_cut.masked_fill(mask_in, 0.)[:, k_diff_i,:] - eig_label_cut.masked_fill(mask_in, 0.)[:,k_diff_j,:]
                    eig_ddiff_pred = eig_pred_cut.masked_fill(mask_in, 0.)[:,k_diff_i,:] - eig_pred_cut.masked_fill(mask_in, 0.)[:,k_diff_j,:]
                else:
                    eig_diff_lbl = eig_label_cut[:,k_diff_i,:] - eig_label_cut[:,k_diff_j,:]
                    eig_ddiff_pred = eig_pred_cut[:,k_diff_i,:]  - eig_pred_cut[:,k_diff_j,:]
                loss_diff =  mse_loss(eig_diff_lbl, eig_ddiff_pred) 
                
                loss = loss + self.diff_weight * loss_diff

            total_loss += loss

        return total_loss / len(datalist)

# @Loss.register("hamil")
# class HamilLoss(nn.Module):
#     def __init__(
#             self, 
#             basis: Dict[str, Union[str, list]]=None,
#             idp: Union[OrbitalMapper, None]=None,
#             overlap: bool=False,
#             dtype: Union[str, torch.dtype] = torch.float32, 
#             device: Union[str, torch.device] = torch.device("cpu"),
#             **kwargs,
#         ):

#         super(HamilLoss, self).__init__()
#         self.loss1 = nn.L1Loss()
#         self.loss2 = nn.MSELoss()
#         self.overlap = overlap
#         self.device = device

#         if basis is not None:
#             self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
#             if idp is not None:
#                 assert idp == self.idp, "The basis of idp and basis should be the same."
#         else:
#             assert idp is not None, "Either basis or idp should be provided."
#             self.idp = idp

#     def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
#         # mask the data

#         # data[AtomicDataDict.NODE_FEATURES_KEY].masked_fill(~self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY]], 0.)
#         # data[AtomicDataDict.EDGE_FEATURES_KEY].masked_fill(~self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY]], 0.)

#         node_mean = ref_data[AtomicDataDict.NODE_FEATURES_KEY].mean(dim=-1, keepdim=True)
#         edge_mean = ref_data[AtomicDataDict.EDGE_FEATURES_KEY].mean(dim=-1, keepdim=True)
#         node_weight = 1/((ref_data[AtomicDataDict.NODE_FEATURES_KEY]-node_mean).norm(dim=-1, keepdim=True)+1e-5)
#         edge_weight = 1/((ref_data[AtomicDataDict.EDGE_FEATURES_KEY]-edge_mean).norm(dim=-1, keepdim=True)+1e-5)
        
#         pre = (node_weight*(data[AtomicDataDict.NODE_FEATURES_KEY]-node_mean))[self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
#         tgt = (node_weight*(ref_data[AtomicDataDict.NODE_FEATURES_KEY]-node_mean))[self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
#         onsite_loss = self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt))

#         pre = (edge_weight*(data[AtomicDataDict.EDGE_FEATURES_KEY]-edge_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
#         tgt = (edge_weight*(ref_data[AtomicDataDict.EDGE_FEATURES_KEY]-edge_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
#         hopping_loss = self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt))
        
#         if self.overlap:
#             over_mean = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY].mean(dim=-1, keepdim=True)
#             over_weight = 1/((ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]-over_mean).norm(dim=-1, keepdim=True)+1e-5)
#             pre = (over_weight*(data[AtomicDataDict.EDGE_OVERLAP_KEY]-over_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
#             tgt = (over_weight*(ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]-over_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
#             hopping_loss += self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt))
        
#         return hopping_loss + onsite_loss

def shift_mu(data: AtomicDataDict, ref_data: AtomicDataDict,idp:OrbitalMapper):
    mu_n = (data[AtomicDataDict.NODE_FEATURES_KEY] - ref_data[AtomicDataDict.NODE_FEATURES_KEY]) * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
    mu_n = mu_n.sum(dim=-1) # [natoms]
    mu_n_diag = (data[AtomicDataDict.NODE_FEATURES_KEY][:,idp.full_mask_to_diag] - 
                    ref_data[AtomicDataDict.NODE_FEATURES_KEY][:,idp.full_mask_to_diag]) * ref_data[AtomicDataDict.NODE_OVERLAP_KEY][:,idp.full_mask_to_diag]
    mu_n_diag = mu_n_diag.sum(dim=-1) # [natoms]
    mu_n_all = mu_n * 2 - mu_n_diag

    mu_e = (data[AtomicDataDict.EDGE_FEATURES_KEY] - ref_data[AtomicDataDict.EDGE_FEATURES_KEY]) * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
    mu_e = mu_e.sum(dim=-1) # [edges]
    mu_e_diag = (data[AtomicDataDict.EDGE_FEATURES_KEY][:,idp.full_mask_to_diag] - 
                    ref_data[AtomicDataDict.EDGE_FEATURES_KEY][:,idp.full_mask_to_diag])  * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][:,idp.full_mask_to_diag] 
    mu_e_diag = mu_e_diag.sum(dim=-1) # [edges]
    mu_e_all = mu_e*2 - mu_e_diag

    norm_ss_n =  (ref_data[AtomicDataDict.NODE_OVERLAP_KEY] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]).sum(dim=-1)
    norm_ss_n_diag = (ref_data[AtomicDataDict.NODE_OVERLAP_KEY][:,idp.full_mask_to_diag] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY][:,idp.full_mask_to_diag]).sum(dim=-1)
    norm_ss_n_all = norm_ss_n * 2 - norm_ss_n_diag
    
    norm_ss_e =  (ref_data[AtomicDataDict.EDGE_OVERLAP_KEY] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]).sum(dim=-1)
    norm_ss_e_diag = (ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][:,idp.full_mask_to_diag] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][:,idp.full_mask_to_diag]).sum(dim=-1)
    norm_ss_e_all = norm_ss_e * 2 - norm_ss_e_diag

    return mu_n_all, mu_e_all, norm_ss_n_all, norm_ss_e_all


def _batch_tensor(data: AtomicDataDict, device: torch.device):
    batch = data.get("batch", None)
    if batch is None:
        return torch.zeros(
            data[AtomicDataDict.POSITIONS_KEY].shape[0],
            dtype=torch.long,
            device=device,
        )
    return batch.to(device=device, dtype=torch.long)


def _slices_for(data: AtomicDataDict, key: str):
    return data.get("__slices__", {}).get(key)


def _edge_graph_index(data: AtomicDataDict, device: torch.device):
    edge_slices = _slices_for(data, AtomicDataDict.EDGE_INDEX_KEY)
    n_edges = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    if edge_slices is None:
        batch = _batch_tensor(data, device)
        return batch[data[AtomicDataDict.EDGE_INDEX_KEY][0]]

    edge_mu_index = torch.empty(n_edges, dtype=torch.long, device=device)
    for graph_idx, (start, end) in enumerate(zip(edge_slices[:-1], edge_slices[1:])):
        edge_mu_index[int(start):int(end)] = graph_idx
    return edge_mu_index


def _apply_shift_mu(data: AtomicDataDict, ref_data: AtomicDataDict, idp: OrbitalMapper):
    device = ref_data[AtomicDataDict.NODE_FEATURES_KEY].device
    batch = _batch_tensor(data, device)
    mu_n, mu_e, norm_ss_n, norm_ss_e = shift_mu(data=data, ref_data=ref_data, idp=idp)
    node_slices = _slices_for(data, AtomicDataDict.POSITIONS_KEY)
    edge_slices = _slices_for(data, AtomicDataDict.EDGE_INDEX_KEY)

    if node_slices is None or edge_slices is None or len(node_slices) <= 2:
        diffhs = mu_n.sum() + mu_e.sum()
        ss = norm_ss_n.sum() + norm_ss_e.sum()
        mu = (diffhs / ss).detach()
        ref_data[AtomicDataDict.NODE_FEATURES_KEY] = (
            ref_data[AtomicDataDict.NODE_FEATURES_KEY]
            + mu * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
        )
        ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = (
            ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
            + mu * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
        )
        return

    mu_n = torch.stack([mu_n[int(node_slices[i]):int(node_slices[i + 1])].sum() for i in range(len(node_slices) - 1)])
    mu_e = torch.stack([mu_e[int(edge_slices[i]):int(edge_slices[i + 1])].sum() for i in range(len(edge_slices) - 1)])

    norm_ss_n = torch.stack([
        norm_ss_n[int(node_slices[i]):int(node_slices[i + 1])].sum()
        for i in range(len(node_slices) - 1)
    ])
    norm_ss_e = torch.stack([
        norm_ss_e[int(edge_slices[i]):int(edge_slices[i + 1])].sum()
        for i in range(len(edge_slices) - 1)
    ])

    mu = ((mu_n + mu_e) / (norm_ss_n + norm_ss_e)).detach()
    ref_data[AtomicDataDict.NODE_FEATURES_KEY] = (
        ref_data[AtomicDataDict.NODE_FEATURES_KEY]
        + mu[batch, None] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
    )
    edge_mu_index = _edge_graph_index(data, device)
    ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = (
        ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
        + mu[edge_mu_index, None] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
    )


def _apply_diag_onsite_shift(data: AtomicDataDict, ref_data: AtomicDataDict, idp: OrbitalMapper):
    device = ref_data[AtomicDataDict.NODE_FEATURES_KEY].device
    batch = _batch_tensor(data, device)
    atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
    diag_mask = _take_idp_tensor(idp.mask_to_ndiag, atom_types, result_device=device)
    diag_weight = diag_mask.to(dtype=ref_data[AtomicDataDict.NODE_FEATURES_KEY].dtype)
    diag_diff = (
        data[AtomicDataDict.NODE_FEATURES_KEY]
        - ref_data[AtomicDataDict.NODE_FEATURES_KEY]
    ) * diag_weight
    node_slices = _slices_for(data, AtomicDataDict.POSITIONS_KEY)

    if node_slices is None or len(node_slices) <= 2:
        denom = diag_weight.sum()
        mu = (diag_diff.sum() / denom).detach()
        ref_data[AtomicDataDict.NODE_FEATURES_KEY] = (
            ref_data[AtomicDataDict.NODE_FEATURES_KEY]
            + mu * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
        )
        ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = (
            ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
            + mu * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
        )
        return

    mu = torch.stack([
        (
            diag_diff[int(node_slices[i]):int(node_slices[i + 1])].sum()
            / diag_weight[int(node_slices[i]):int(node_slices[i + 1])].sum()
        )
        for i in range(len(node_slices) - 1)
    ]).detach()
    ref_data[AtomicDataDict.NODE_FEATURES_KEY] = (
        ref_data[AtomicDataDict.NODE_FEATURES_KEY]
        + mu[batch, None] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
    )
    edge_mu_index = _edge_graph_index(data, device)
    ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = (
        ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
        + mu[edge_mu_index, None] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
    )


@Loss.register("eig_ham")
class EigHamLoss(nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            onsite_shift: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            diff_on: bool=False,
            eout_weight: float=0.01,
            diff_weight: float=0.01,
            diff_valence: dict=None,
            spin_deg: int = 2,
            coeff_ham: float=1.,
            coeff_ovp: float=1.,
            **kwargs,
        ):
        super(EigHamLoss, self).__init__()
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.MSELoss()
        self.overlap = overlap
        self.device = device
        self.onsite_shift = onsite_shift
        self.coeff_ham = coeff_ham
        assert self.coeff_ham <= 1.
        self.coeff_ovp = coeff_ovp

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.eigloss = EigLoss(
            idp=self.idp,
            overlap=overlap,
            diff_on=diff_on,
            eout_weight=eout_weight,
            diff_weight=diff_weight,
            diff_valence=diff_valence,
            spin_deg=spin_deg,
            dtype=dtype, 
            device=device,
        )

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        # mask the data

        if self.onsite_shift:
            _apply_shift_mu(data, ref_data, self.idp)
                
        pre = data[AtomicDataDict.NODE_FEATURES_KEY][_nrme_mask(
            self.idp,
            data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            result_device=data[AtomicDataDict.NODE_FEATURES_KEY].device,
        )]
        tgt = ref_data[AtomicDataDict.NODE_FEATURES_KEY][_nrme_mask(
            self.idp,
            ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            result_device=ref_data[AtomicDataDict.NODE_FEATURES_KEY].device,
        )]
        onsite_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

        pre = data[AtomicDataDict.EDGE_FEATURES_KEY][_erme_mask(
            self.idp,
            data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            result_device=data[AtomicDataDict.EDGE_FEATURES_KEY].device,
        )]
        tgt = ref_data[AtomicDataDict.EDGE_FEATURES_KEY][_erme_mask(
            self.idp,
            ref_data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            result_device=ref_data[AtomicDataDict.EDGE_FEATURES_KEY].device,
        )]
        hopping_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))
        
        if self.overlap:
            pre = data[AtomicDataDict.EDGE_OVERLAP_KEY][_erme_mask(
                self.idp,
                data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                result_device=data[AtomicDataDict.EDGE_OVERLAP_KEY].device,
            )]
            tgt = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][_erme_mask(
                self.idp,
                ref_data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                result_device=ref_data[AtomicDataDict.EDGE_OVERLAP_KEY].device,
            )]
            overlap_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

            pre = data[AtomicDataDict.NODE_OVERLAP_KEY][_nrme_mask(
                self.idp,
                data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                result_device=data[AtomicDataDict.NODE_OVERLAP_KEY].device,
            )]
            tgt = ref_data[AtomicDataDict.NODE_OVERLAP_KEY][_nrme_mask(
                self.idp,
                ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                result_device=ref_data[AtomicDataDict.NODE_OVERLAP_KEY].device,
            )]
            overlap_loss += 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

            ham_loss = (1/3) * (hopping_loss + onsite_loss + (self.coeff_ovp / self.coeff_ham) * overlap_loss)
        else:
            ham_loss = 0.5 * (onsite_loss + hopping_loss)

        eigloss = self.eigloss(data, ref_data)

        return self.coeff_ham * ham_loss + (1 - self.coeff_ham) * eigloss


import logging
import inspect
import torch

# 假设这些类已经在上下文或其他地方定义
# from somewhere import Loss, OrbitalMapper, AtomicDataDict

log = logging.getLogger(__name__)

import torch.nn as nn
from typing import Dict, Union
import math


@Loss.register("hamil_abs")
class HamilLossAbs(nn.Module):
    supports_endpoint_triplet = True
    endpoint_metric_space = "rme"

    def __init__(
            self,
            basis: Dict[str, Union[str, list]] = None,
            idp: Union[OrbitalMapper, None] = None,
            overlap: bool = False,
            onsite_shift: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            debug_flag: bool = False,
            nextham_uureal_mask: bool = False,
            onsite_boost: bool = False,
            onsite_boost_steps: int = 20000,
            onsite_boost_max: float = 100.0,
            z_loss_coef: float = 0.0,
            element_average: bool = False,
            **kwargs,
    ):
        super(HamilLossAbs, self).__init__()
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.MSELoss()
        self.overlap = overlap
        self.device = device
        self.onsite_shift = onsite_shift
        self.dtype = dtype

        self.debug = debug_flag
        self._debug_counter = 0
        self.z_loss_coef = float(z_loss_coef)
        self.last_z_loss = None
        self.expert_load_cv = None
        self.element_average = bool(element_average)

        self.onsite_boost = bool(onsite_boost)
        self.onsite_boost_steps = int(onsite_boost_steps)
        self.onsite_boost_max = float(onsite_boost_max)
        self._step = 0
        self.last_onsite_loss = None
        self.last_hopping_loss = None

        if self.debug:
            self._log_caller_info(kwargs)

        full_soc_prediction = bool(kwargs.get("full_soc_prediction", False))
        nextham_uureal_mask = resolve_nextham_uureal_mask(
            nextham_uureal_mask=nextham_uureal_mask,
            full_soc_prediction=full_soc_prediction,
        )

        if basis is not None:
            has_soc = kwargs.get('has_soc', False)
            self.idp = OrbitalMapper(
                basis,
                method="e3tb",
                device=self.device,
                has_soc=has_soc,
                nextham_uureal_mask=nextham_uureal_mask,
                full_soc_prediction=full_soc_prediction,
            )
            log.warning(f'initialize loss rme with nextham_uureal_mask: {nextham_uureal_mask}')
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

    def _current_onsite_weight(self) -> float:
        if not self.onsite_boost:
            return 1.0
        progress = min(self._step / max(self.onsite_boost_steps, 1), 1.0)
        return self.onsite_boost_max - (self.onsite_boost_max - 1.0) * progress

    def compatible_loss_from_stats(
        self,
        *,
        onsite_l1_sum: torch.Tensor,
        onsite_mse_sum: torch.Tensor,
        onsite_count: torch.Tensor,
        hopping_l1_sum: torch.Tensor,
        hopping_mse_sum: torch.Tensor,
        hopping_count: torch.Tensor,
        z_loss=None,
        global_step=None,
    ):
        """Rebuild the non-CFM clean HamilLossAbs values from reduced stats."""
        if global_step is not None:
            self._step = int(global_step)

        onsite_loss = _l1_rmse_loss_from_sums(
            abs_sum=onsite_l1_sum,
            square_sum=onsite_mse_sum,
            count=onsite_count,
        )
        hopping_loss = _l1_rmse_loss_from_sums(
            abs_sum=hopping_l1_sum,
            square_sum=hopping_mse_sum,
            count=hopping_count,
        )

        self.last_onsite_l1_sum = onsite_l1_sum.detach()
        self.last_onsite_mse_sum = onsite_mse_sum.detach()
        self.last_onsite_count = onsite_count.detach()
        self.last_hopping_l1_sum = hopping_l1_sum.detach()
        self.last_hopping_mse_sum = hopping_mse_sum.detach()
        self.last_hopping_count = hopping_count.detach()
        self.last_onsite_loss = onsite_loss.detach()
        self.last_hopping_loss = hopping_loss.detach()

        if self.onsite_boost:
            total_loss = self._current_onsite_weight() * onsite_loss + hopping_loss
        elif self.element_average:
            total_loss = _l1_rmse_loss_from_sums(
                abs_sum=onsite_l1_sum + hopping_l1_sum,
                square_sum=onsite_mse_sum + hopping_mse_sum,
                count=onsite_count + hopping_count,
            )
        else:
            total_loss = 0.5 * (onsite_loss + hopping_loss)

        self.last_z_loss = z_loss.detach() if isinstance(z_loss, torch.Tensor) else z_loss
        if self.z_loss_coef > 0 and isinstance(z_loss, torch.Tensor):
            total_loss = total_loss + self.z_loss_coef * z_loss

        return total_loss.detach(), onsite_loss.detach(), hopping_loss.detach()

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        if "global_step" in data:
            try:
                self._step = int(data["global_step"])
            except Exception:
                self._step = self._step + 1
        else:
            self._step += 1

        self._debug_counter += 1
        verbose_step = self.debug and (self._debug_counter <= 1 or self._debug_counter % 5 == 1)

        try:
            # =================================================================
            # Onsite (纯张量流：彻底消灭 .any() 与索引操作带来的隐式 CUDA 同步)
            # =================================================================
            atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
            node_mask_orb = _nrme_mask(
                self.idp,
                atom_types,
                result_device=data[AtomicDataDict.NODE_FEATURES_KEY].device,
            )  # (N, 107)

            if "expert_node_mask" in data:
                node_mask_phy = data["expert_node_mask"].unsqueeze(-1)  # (N,1)
                final_node_mask = node_mask_orb & node_mask_phy
            else:
                final_node_mask = node_mask_orb

            raw_pre_node = data[AtomicDataDict.NODE_FEATURES_KEY]
            raw_tgt_node = ref_data[AtomicDataDict.NODE_FEATURES_KEY]
            raw_pre_node, _raw_mask = project_uureal_to_like(self.idp, raw_pre_node, raw_tgt_node)
            if raw_pre_node.shape != raw_tgt_node.shape:
                raise ValueError(
                    "node prediction layout does not match target layout; "
                    "check nextham_uureal_mask/mask_uureal propagation."
                )
            final_node_mask = normalize_idp_mask_layout(
                self.idp,
                final_node_mask,
                raw_tgt_node,
                label="node idp mask",
            )

            diff_node = (raw_pre_node - raw_tgt_node) * final_node_mask
            abs_node = diff_node.abs()
            sq_node = diff_node * diff_node

            onsite_l1_sum = abs_node.sum()
            onsite_mse_sum = sq_node.sum()

            # 使用 raw_pre_node 自身的 dtype
            onsite_cnt = final_node_mask.sum().to(dtype=raw_pre_node.dtype)
            onsite_loss = _l1_rmse_loss_from_sums(
                abs_sum=onsite_l1_sum,
                square_sum=onsite_mse_sum,
                count=onsite_cnt,
            )

            # =================================================================
            # Hopping (同理重构)
            # =================================================================
            edge_types = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()
            edge_mask_orb = _erme_mask(
                self.idp,
                edge_types,
                result_device=data[AtomicDataDict.EDGE_FEATURES_KEY].device,
            )  # (E, 128)

            if "expert_edge_mask" in data:
                edge_mask_phy = data["expert_edge_mask"].unsqueeze(-1)  # (E,1)
                final_edge_mask = edge_mask_orb & edge_mask_phy
            else:
                final_edge_mask = edge_mask_orb

            raw_pre_edge = data[AtomicDataDict.EDGE_FEATURES_KEY]
            raw_tgt_edge = ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
            raw_pre_edge, _raw_mask = project_uureal_to_like(self.idp, raw_pre_edge, raw_tgt_edge)
            if raw_pre_edge.shape != raw_tgt_edge.shape:
                raise ValueError(
                    "edge prediction layout does not match target layout; "
                    "check nextham_uureal_mask/mask_uureal propagation."
                )
            final_edge_mask = normalize_idp_mask_layout(
                self.idp,
                final_edge_mask,
                raw_tgt_edge,
                label="edge idp mask",
            )

            diff_edge = (raw_pre_edge - raw_tgt_edge) * final_edge_mask
            abs_edge = diff_edge.abs()
            sq_edge = diff_edge * diff_edge

            hopping_l1_sum = abs_edge.sum()
            hopping_mse_sum = sq_edge.sum()

            # 使用 raw_pre_edge 自身的 dtype
            hopping_cnt = final_edge_mask.sum().to(dtype=raw_pre_edge.dtype)
            hopping_loss = _l1_rmse_loss_from_sums(
                abs_sum=hopping_l1_sum,
                square_sum=hopping_mse_sum,
                count=hopping_cnt,
            )

            # ========== record strict reduce stats ==========
            self.last_onsite_l1_sum = onsite_l1_sum.detach()
            self.last_onsite_mse_sum = onsite_mse_sum.detach()
            self.last_onsite_count = onsite_cnt.detach()

            self.last_hopping_l1_sum = hopping_l1_sum.detach()
            self.last_hopping_mse_sum = hopping_mse_sum.detach()
            self.last_hopping_count = hopping_cnt.detach()

            # ========== existing metrics ==========
            raw_z_loss = data.get("mean_max_prob", 0)
            self.last_z_loss = raw_z_loss.detach() if isinstance(raw_z_loss, torch.Tensor) else raw_z_loss

            expert_load_cv = data.get("expert_load_cv", 0)
            self.expert_load_cv = expert_load_cv.detach() if isinstance(expert_load_cv,
                                                                        torch.Tensor) else expert_load_cv

            self.last_onsite_loss = onsite_loss.detach()
            self.last_hopping_loss = hopping_loss.detach()

            if verbose_step:
                self._log_step_info(locals())
                if self.onsite_boost:
                    print(f"  Onsite weight w(t) = {self._current_onsite_weight():.3f}")

            # ========== total ==========
            if self.onsite_boost:
                w_onsite = self._current_onsite_weight()
                total_loss = w_onsite * onsite_loss + hopping_loss
            elif self.element_average:
                total_loss = _l1_rmse_loss_from_sums(
                    abs_sum=onsite_l1_sum + hopping_l1_sum,
                    square_sum=onsite_mse_sum + hopping_mse_sum,
                    count=onsite_cnt + hopping_cnt,
                )
            else:
                total_loss = 0.5 * (onsite_loss + hopping_loss)

            if self.z_loss_coef > 0 and isinstance(raw_z_loss, torch.Tensor):
                total_loss = total_loss + self.z_loss_coef * raw_z_loss

            return total_loss

        except Exception as e:
            if self.debug:
                self._print_crash_info(e, data, locals())
            raise

    # =========================================================================
    #                               Debug Helpers
    # =========================================================================
    def _log_caller_info(self, kwargs):
        log.warning('=' * 44)
        log.warning('========== HamilLossAbs Initialized ==========')
        log.warning(f"KWARGS: {kwargs}")
        log.warning(f"HAS_SOC: {kwargs.get('has_soc', 'Not Provided')}")
        try:
            caller_frame_info = inspect.getouterframes(inspect.currentframe(), 2)[1]
            caller_file = caller_frame_info.filename
            caller_line = caller_frame_info.lineno
            caller_func = caller_frame_info.function
            arg_info = inspect.getargvalues(caller_frame_info.frame)
            caller_class = ""
            if 'self' in arg_info.locals:
                caller_class = arg_info.locals['self'].__class__.__name__ + "."
            log.warning(f"CALLED BY: {caller_class}{caller_func}")
            log.warning(f"LOCATION : {caller_file}:{caller_line}")
        except Exception:
            log.warning("Could not trace caller frame.")
        log.warning('=' * 44)

    def _log_step_info(self, locs):
        print(f"\n[Loss Debug] Step {self._debug_counter}")
        if 'raw_pre_node' in locs and 'raw_tgt_node' in locs:
            print(f"  Node Raw Shapes -> Pre: {locs['raw_pre_node'].shape}, Tgt: {locs['raw_tgt_node'].shape}")
        if 'final_node_mask' in locs:
            print(
                f"  Node Mask Final -> Shape: {locs['final_node_mask'].shape}, Selected: {locs['final_node_mask'].sum().item()}")
        if 'raw_pre_edge' in locs and 'raw_tgt_edge' in locs:
            print(f"  Edge Raw Shapes -> Pre: {locs['raw_pre_edge'].shape}, Tgt: {locs['raw_tgt_edge'].shape}")
        if 'final_edge_mask' in locs:
            print(
                f"  Edge Mask Final -> Shape: {locs['final_edge_mask'].shape}, Selected: {locs['final_edge_mask'].sum().item()}")
        if 'onsite_loss' in locs and 'hopping_loss' in locs:
            print(
                f"  Loss Values -> Onsite: {locs['onsite_loss'].item():.6f}, Hopping: {locs['hopping_loss'].item():.6f}")

    def _print_crash_info(self, error, data, locs):
        print(f"\n{'!' * 20} Loss Forward Crash {'!' * 20}")
        print(f"Error: {str(error)}")
        print(f"Step: {self._debug_counter}")
        try:
            print(f"Atom Types Shape: {data[AtomicDataDict.ATOM_TYPE_KEY].shape}")
            print(f"Edge Types Shape: {data[AtomicDataDict.EDGE_TYPE_KEY].shape}")
        except:
            print("Could not access input data shapes.")
        vars_to_check = ['final_node_mask', 'final_edge_mask', 'raw_pre_node', 'raw_tgt_node', 'raw_pre_edge',
                         'raw_tgt_edge']
        for var_name in vars_to_check:
            if var_name in locs:
                tensor = locs[var_name]
                info = f"Sum: {tensor.sum()}" if tensor.dtype == torch.bool else "Feature Tensor"
                print(f"{var_name}: Shape {tensor.shape}, {info}")
        print('!' * 60)


@Loss.register("hamil_blas")
class HamilLossBlas(nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            onsite_shift: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLossBlas, self).__init__()
        self.overlap = overlap
        self.device = device
        self.onsite_shift = onsite_shift

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        # mask the data
        if self.onsite_shift:
            _apply_shift_mu(data, ref_data, self.idp)
                
        onsite_loss = data[AtomicDataDict.NODE_FEATURES_KEY]-ref_data[AtomicDataDict.NODE_FEATURES_KEY]
        onsite_index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().unique()
        onsite_loss = scatter_mean(
            src = onsite_loss.abs(), 
            index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.type_names)
            )[onsite_index][_nrme_mask(self.idp, onsite_index, result_device=data[AtomicDataDict.NODE_FEATURES_KEY].device)].mean() + scatter_mean(
            src = onsite_loss**2,
            index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.type_names)
        )[onsite_index][_nrme_mask(self.idp, onsite_index, result_device=data[AtomicDataDict.NODE_FEATURES_KEY].device)].mean().sqrt()
        onsite_loss *= 0.5

        hopping_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten().unique()
        hopping_loss = data[AtomicDataDict.EDGE_FEATURES_KEY]-ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
        hopping_loss = scatter_mean(
            src = hopping_loss.abs(), 
            index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.bond_types)
            )[hopping_index][_erme_mask(self.idp, hopping_index, result_device=data[AtomicDataDict.EDGE_FEATURES_KEY].device)].mean() + scatter_mean(
            src = hopping_loss**2,
            index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.bond_types)
        )[hopping_index][_erme_mask(self.idp, hopping_index, result_device=data[AtomicDataDict.EDGE_FEATURES_KEY].device)].mean().sqrt()
        hopping_loss *= 0.5
        
        if self.overlap:
            overlap_loss = data[AtomicDataDict.EDGE_OVERLAP_KEY]-ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            overlap_loss = scatter_mean(
                src = overlap_loss.abs(), 
                index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.bond_types)
                )[hopping_index][_erme_mask(self.idp, hopping_index, result_device=data[AtomicDataDict.EDGE_OVERLAP_KEY].device)].mean() + scatter_mean(
                src = overlap_loss**2,
                index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.bond_types)
            )[hopping_index][_erme_mask(self.idp, hopping_index, result_device=data[AtomicDataDict.EDGE_OVERLAP_KEY].device)].mean().sqrt()
            overlap_loss *= 0.5

            overlap_onsite_loss = data[AtomicDataDict.NODE_OVERLAP_KEY]-ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
            overlap_onsite_loss = scatter_mean(
                src = overlap_onsite_loss.abs(), 
                index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.type_names)
                )[onsite_index][_nrme_mask(self.idp, onsite_index, result_device=data[AtomicDataDict.NODE_OVERLAP_KEY].device)].mean() + scatter_mean(
                src = overlap_onsite_loss**2,
                index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.type_names)
            )[onsite_index][_nrme_mask(self.idp, onsite_index, result_device=data[AtomicDataDict.NODE_OVERLAP_KEY].device)].mean().sqrt()
            overlap_loss += overlap_onsite_loss * 0.5

            return (1/3) * (hopping_loss + onsite_loss + overlap_loss)
        else:
            return 0.5 * (onsite_loss + hopping_loss)


@Loss.register("hamil_abs_element_avg")
class HamilLossAbsElementAverage(HamilLossAbs):
    def __init__(self, *args, **kwargs):
        kwargs["element_average"] = True
        super().__init__(*args, **kwargs)


@Loss.register("hamil_abs_mae")
class HamilLossAbsMAE(nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]] = None,
            idp: Union[OrbitalMapper, None] = None,
            overlap: bool = False,
            onsite_shift: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            # ===== 新增参数（默认不启用） =====
            onsite_boost: bool = False,
            onsite_boost_steps: int = 20000,
            onsite_boost_max: float = 100.0,
            **kwargs,
    ):

        super(HamilLossAbsMAE, self).__init__()
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.MSELoss()
        self.overlap = overlap
        self.device = device
        self.onsite_shift = onsite_shift

        # 新增：onsite 动态加权控制
        self.onsite_boost = bool(onsite_boost)
        self.onsite_boost_steps = int(onsite_boost_steps)
        self.onsite_boost_max = float(onsite_boost_max)
        self._step = 0  # 内部迭代计数器

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

    def _current_onsite_weight(self) -> float:
        """
        从 0 到 onsite_boost_steps 线性衰减：
            step=0       => weight = onsite_boost_max
            step>=steps  => weight = 1.0
        """
        if not self.onsite_boost:
            return 1.0
        progress = min(self._step / max(self.onsite_boost_steps, 1), 1.0)
        w = self.onsite_boost_max - (self.onsite_boost_max - 1.0) * progress
        return float(w)

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        # ------------- onsite_shift 原逻辑保持不动 -------------
        if self.onsite_shift:
            _apply_diag_onsite_shift(data, ref_data, self.idp)

        # ------------- 取出被 mask 的 onsite / hopping 特征 -------------
        pre_onsite = data[AtomicDataDict.NODE_FEATURES_KEY][
            _nrme_mask(
                self.idp,
                data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                result_device=data[AtomicDataDict.NODE_FEATURES_KEY].device,
            )
        ]
        tgt_onsite = ref_data[AtomicDataDict.NODE_FEATURES_KEY][
            _nrme_mask(
                self.idp,
                ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                result_device=ref_data[AtomicDataDict.NODE_FEATURES_KEY].device,
            )
        ]

        pre_hopping = data[AtomicDataDict.EDGE_FEATURES_KEY][
            _erme_mask(
                self.idp,
                data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                result_device=data[AtomicDataDict.EDGE_FEATURES_KEY].device,
            )
        ]
        tgt_hopping = ref_data[AtomicDataDict.EDGE_FEATURES_KEY][
            _erme_mask(
                self.idp,
                ref_data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                result_device=ref_data[AtomicDataDict.EDGE_FEATURES_KEY].device,
            )
        ]

        # ------------- 新增：动态加权 onsite -------------
        if self.onsite_boost:
            w_onsite = self._current_onsite_weight()
            loss_onsite = self.loss1(pre_onsite, tgt_onsite)
            loss_hopping = self.loss1(pre_hopping, tgt_hopping)
            total_loss = w_onsite * loss_onsite + loss_hopping
            self._step += 1
            return total_loss

        # ------------- 旧逻辑：不区分 onsite/hopping 权重 -------------
        pre = torch.cat([pre_onsite, pre_hopping], dim=0)
        tgt = torch.cat([tgt_onsite, tgt_hopping], dim=0)


        total_loss = self.loss1(pre, tgt)
        return total_loss


def get_electron_number_from_dm_torch(dm, overlap):
    """
    [修复] 使用 PyTorch 操作替代 NumPy，确保:
    1. 梯度可以回传 (Gradient Flow)
    2. 数据均在 GPU 上
    3. 类型匹配
    """
    P = dm
    S = overlap

    # 确保 overlap 类型与 P 一致 (float)
    if P.dtype != S.dtype:
        S = S.to(P.dtype)

    # 如果是自旋分辨的 3D DM (spin, nao, nao)，则先对自旋求和
    if P.ndim == 3:
        P = torch.sum(P, dim=0)

    # 如果 Overlap 是 3D
    if S.ndim == 3:
        S = S[0]

    # [关键修复] 使用 torch.einsum 替代 np.einsum
    # 计算 Trace(P @ S)
    return torch.einsum("ij,ji->", P, S)


@Loss.register("hamil_w_num_e")
class HamilNumE(nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]] = None,
            idp: Union[OrbitalMapper, None] = None,
            overlap: bool = False,
            on_the_fly_ovp_flag: bool = False,
            num_e_loss_weight: float = 0.1,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
    ):
        super(HamilNumE, self).__init__()
        self.loss1 = nn.L1Loss()  # 用于 Hamiltonian 和 电子数 Loss
        self.loss2 = nn.MSELoss()
        self.overlap = overlap
        self.device = device
        self.on_the_fly_ovp_flag = on_the_fly_ovp_flag
        self.num_e_loss_weight = num_e_loss_weight

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.dm_hr2hk = HR2HK_Gamma_Only(
            idp=self.idp,
            edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            node_field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.HAMILTONIAN_KEY,
            device=device
        )
        self.overlap_hr2hk = HR2HK_Gamma_Only(
            idp=self.idp,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            device=device
        )
        self.kpoint = torch.tensor([[0.0, 0.0, 0.0]], device=device)


    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        data['kpoint'] = self.kpoint
        ref_data['kpoint'] = self.kpoint

        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        # atomic_numbers 是 int/long 类型
        atomic_numbers = self.idp.untransform_atom(type_numbers)
        #######################################################################################
        #######################################################################################

        # 预测 DM (Float, on CUDA, requires_grad=True)
        pred_dm_data = self.dm_hr2hk.forward(data)
        pred_dm = pred_dm_data[AtomicDataDict.HAMILTONIAN_KEY]

        # 真实 Overlap (Float, on CUDA)
        ref_overlap_data = self.overlap_hr2hk.forward(ref_data)
        ref_overlap = ref_overlap_data[AtomicDataDict.OVERLAP_KEY]

        # 计算预测电子数 (Float, Tensor)
        pred_electron_number = get_electron_number_from_dm_torch(pred_dm, ref_overlap)

        # 获取真实电荷 (Int, on CUDA)
        real_charge = ref_data['charge']

        # 计算目标总电子数 (Int)
        # sum_of_atomic_numbers (Int) - real_charge (Int) = target_electrons (Int)
        sum_of_atomic_numbers = atomic_numbers.sum()
        total_electrons = sum_of_atomic_numbers - real_charge

        electron_number_loss = self.loss1(pred_electron_number, total_electrons.float())

        #######################################################################################

        # onsite loss
        pre_onsite = data[AtomicDataDict.NODE_FEATURES_KEY][
            _nrme_mask(
                self.idp,
                data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                result_device=data[AtomicDataDict.NODE_FEATURES_KEY].device,
            )
        ]
        tgt_onsite = ref_data[AtomicDataDict.NODE_FEATURES_KEY][
            _nrme_mask(
                self.idp,
                ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                result_device=ref_data[AtomicDataDict.NODE_FEATURES_KEY].device,
            )
        ]
        # hopping loss
        pre_hopping = data[AtomicDataDict.EDGE_FEATURES_KEY][
            _erme_mask(
                self.idp,
                data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                result_device=data[AtomicDataDict.EDGE_FEATURES_KEY].device,
            )
        ]
        tgt_hopping = ref_data[AtomicDataDict.EDGE_FEATURES_KEY][
            _erme_mask(
                self.idp,
                ref_data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                result_device=ref_data[AtomicDataDict.EDGE_FEATURES_KEY].device,
            )
        ]

        pre = torch.cat([pre_onsite, pre_hopping], dim=0)
        tgt = torch.cat([tgt_onsite, tgt_hopping], dim=0)

        hamil_loss = self.loss1(pre, tgt)

        # 加权求和
        final_loss = hamil_loss + self.num_e_loss_weight * electron_number_loss

        return final_loss

@Loss.register("hamil_wt")
class HamilLossWT(nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            onsite_shift: bool=False,
            onsite_weight: Union[float, int, dict]=1.,
            hopping_weight: Union[float, int, dict]=1.,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLossWT, self).__init__()
        self.overlap = overlap
        self.device = device
        self.onsite_shift = onsite_shift

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.onsite_weight = torch.ones(idp.num_types)
        self.hopping_weight = torch.ones(len(idp.bond_types))
        if isinstance(onsite_weight, float) or isinstance(onsite_weight, int):
            self.onsite_weight *= onsite_weight
        elif isinstance(onsite_weight, dict):
            for k,v in onsite_weight.items():
                self.onsite_weight[idp.chemical_symbol_to_type[k]] = v
        else:
            raise TypeError("onsite weight should be either float, int or dict")
        
        if isinstance(hopping_weight, float) or isinstance(hopping_weight, int):
            self.hopping_weight *= hopping_weight
        elif isinstance(hopping_weight, dict):
            for k,v in hopping_weight.items():
                self.hopping_weight[idp.bond_to_type[k]] = v
        else:
            raise TypeError("hopping weight should be either float, int or dict")
        
        self.onsite_weight = self.onsite_weight.unsqueeze(1)
        self.hopping_weight = self.hopping_weight.unsqueeze(1)

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        # mask the data

        if self.onsite_shift:
            _apply_shift_mu(data, ref_data, self.idp)
                
        onsite_loss = data[AtomicDataDict.NODE_FEATURES_KEY]-ref_data[AtomicDataDict.NODE_FEATURES_KEY]
        onsite_index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().unique()
        onsite_loss = (self.onsite_weight * scatter_mean(
            src = onsite_loss.abs(), 
            index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.type_names)
            )[onsite_index])[_nrme_mask(self.idp, onsite_index, result_device=data[AtomicDataDict.NODE_FEATURES_KEY].device)].mean() + (self.onsite_weight**2 * scatter_mean(
            src = onsite_loss**2,
            index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.type_names)
        )[onsite_index])[_nrme_mask(self.idp, onsite_index, result_device=data[AtomicDataDict.NODE_FEATURES_KEY].device)].mean().sqrt()
        onsite_loss *= 0.5

        hopping_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten().unique()
        hopping_loss = data[AtomicDataDict.EDGE_FEATURES_KEY]-ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
        hopping_loss = (self.hopping_weight * scatter_mean(
            src = hopping_loss.abs(), 
            index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.bond_types)
            )[hopping_index])[_erme_mask(self.idp, hopping_index, result_device=data[AtomicDataDict.EDGE_FEATURES_KEY].device)].mean() + (self.hopping_weight**2 * scatter_mean(
            src = hopping_loss**2,
            index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.bond_types)
        )[hopping_index])[_erme_mask(self.idp, hopping_index, result_device=data[AtomicDataDict.EDGE_FEATURES_KEY].device)].mean().sqrt()
        hopping_loss *= 0.5
        
        if self.overlap:
            overlap_loss = data[AtomicDataDict.EDGE_OVERLAP_KEY]-ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            overlap_loss = (self.hopping_weight * scatter_mean(
                src = overlap_loss.abs(), 
                index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.bond_types)
                )[hopping_index])[_erme_mask(self.idp, hopping_index, result_device=data[AtomicDataDict.EDGE_OVERLAP_KEY].device)].mean() + (self.hopping_weight **2 * scatter_mean(
                src = overlap_loss**2,
                index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.bond_types)
            )[hopping_index])[_erme_mask(self.idp, hopping_index, result_device=data[AtomicDataDict.EDGE_OVERLAP_KEY].device)].mean().sqrt()
            overlap_loss *= 0.5

            overlap_onsite_loss = data[AtomicDataDict.NODE_OVERLAP_KEY]-ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
            overlap_onsite_loss = (self.onsite_weight * scatter_mean(
                src = overlap_onsite_loss.abs(), 
                index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.type_names)
                )[onsite_index])[_nrme_mask(self.idp, onsite_index, result_device=data[AtomicDataDict.NODE_OVERLAP_KEY].device)].mean() + ((self.onsite_weight ** 2) * scatter_mean(
                src = overlap_onsite_loss**2,
                index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.type_names)
            )[onsite_index])[_nrme_mask(self.idp, onsite_index, result_device=data[AtomicDataDict.NODE_OVERLAP_KEY].device)].mean().sqrt()
            overlap_loss += overlap_onsite_loss * 0.5

            return (1/3) * (hopping_loss + onsite_loss + overlap_loss)
        else:
            return 0.5 * (onsite_loss + hopping_loss)
        

class HamilLossAnalysis(object):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32,
            decompose: bool = False,
            onsite_shift: bool=False,
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLossAnalysis, self).__init__()
        self.overlap = overlap
        self.device = device
        self.decompose = decompose
        self.dtype = dtype
        self.device = device
        self.onsite_shift = onsite_shift

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.idp.get_irreps()

        if decompose:
            self.e3h = E3Hamiltonian(idp=self.idp, decompose=decompose, overlap=False, device=device, dtype=dtype)
            self.e3s = E3Hamiltonian(idp=self.idp, decompose=decompose, overlap=True, device=device, dtype=dtype)
    
    def __call__(self, data: AtomicDataDict, ref_data: AtomicDataDict, running_avg: bool=False):

        if self.onsite_shift:
            _apply_shift_mu(data, ref_data, self.idp)
        
        for key in ["__slices__", "__cumsum__", "__cat_dims__", "__num_nodes_list__", "__data_class__"]:
            data.pop(key, None)
            ref_data.pop(key, None)

        if self.decompose:
            data = self.e3h(data)
            ref_data = self.e3h(ref_data)
            if self.overlap:
                data = self.e3s(data)
                ref_data = self.e3s(ref_data)
        
        if not running_avg or not hasattr(self, "stats"):
            self.stats = {}
            self.stats["mae"] = 0.
            self.stats["rmse"] = 0.
            self.stats["n_element"] = 0

            # init the self.stats
            self.stats.setdefault("onsite", {})
            self.stats.setdefault("hopping", {})
            if self.overlap:
                self.stats.setdefault("overlap", {})

            for at, tp in self.idp.chemical_symbol_to_type.items():
                self.stats["onsite"][at] = {
                    "rmse":0.,
                    "mae":0.,
                    "rmse_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device), 
                    "mae_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "rmse_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "mae_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "n_element":0,
                }
            
            for bt, tp in self.idp.bond_to_type.items():
                self.stats["hopping"][bt] = {
                    "rmse":0.,
                    "mae":0.,
                    "rmse_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device), 
                    "mae_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "rmse_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "mae_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "n_element":0,
                }

                if self.overlap:
                    self.stats["overlap"][bt] = {
                        "rmse":0.,
                        "mae":0.,
                        "rmse_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device), 
                        "mae_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
                        "rmse_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                        "mae_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                        "n_element":0,
                    }
                
        
        with torch.no_grad():
            n_total = 0
            err = data[AtomicDataDict.NODE_FEATURES_KEY] - ref_data[AtomicDataDict.NODE_FEATURES_KEY]
            mask = self.idp.mask_to_nrme
            onsite = self.stats.get("onsite")
            for at, tp in self.idp.chemical_symbol_to_type.items():
                onsite_mask = mask[tp]
                onsite_err = err[data["atom_types"].flatten().eq(tp)]
                if onsite_err.shape[0] == 0:
                    continue
                onsite_err = onsite_err[:, onsite_mask] # [N_atom_i, n_element]

                rmserr = (onsite_err**2).mean(dim=0).sqrt()
                maerr = onsite_err.abs().mean(dim=0)
                rmse_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                rmse_per_irreps[onsite_mask] = rmserr
                maerr_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                maerr_per_irreps[onsite_mask] = maerr

                rmse_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, rmse_per_irreps)
                maerr_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, maerr_per_irreps)
                
                n_element_old = onsite[at]["n_element"]
                n_total += n_element_old + onsite_err.numel()
                ratio = n_element_old / (n_element_old + onsite_err.numel())
                onsite[at] = {
                    "rmse": ((onsite[at]["rmse"]**2) * ratio + (rmserr**2).mean() * (1-ratio)).sqrt(),
                    "mae":onsite[at]["mae"] * ratio + maerr.mean() * (1-ratio),
                    "rmse_per_block_element": ((onsite[at]["rmse_per_block_element"]**2) * ratio + rmserr**2 * (1-ratio)).sqrt(),
                    "mae_per_block_element": onsite[at]["mae_per_block_element"]*ratio + maerr * (1-ratio),
                    "rmse_per_irreps": ((onsite[at]["rmse_per_irreps"]**2) * ratio + rmse_per_irreps**2 * (1-ratio)).sqrt(),
                    "mae_per_irreps": onsite[at]["mae_per_irreps"] * ratio + maerr_per_irreps * (1-ratio),
                    "n_element":n_element_old + onsite_err.numel(), 
                    }
                
                self.stats["mae"] += onsite[at]["mae"] * onsite[at]["n_element"]
                self.stats["rmse"] += onsite[at]["rmse"]**2 * onsite[at]["n_element"]

            err = data[AtomicDataDict.EDGE_FEATURES_KEY] - ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
            amp = ref_data[AtomicDataDict.EDGE_FEATURES_KEY].abs()
            mask = self.idp.mask_to_erme
            hopping = self.stats.get("hopping", {})
            
            for bt, tp in self.idp.bond_to_type.items():
                hopping_mask = mask[tp]
                hopping_err = err[data["edge_type"].flatten().eq(tp)]
                if hopping_err.shape[0] == 0:
                    continue
                hopping_err = hopping_err[:, hopping_mask]
                
                rmserr = (hopping_err**2).mean(dim=0).sqrt()
                maerr = hopping_err.abs().mean(dim=0)
                rmse_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                rmse_per_irreps[hopping_mask] = rmserr
                maerr_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                maerr_per_irreps[hopping_mask] = maerr

                rmse_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, rmse_per_irreps)
                maerr_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, maerr_per_irreps)
                
                n_element_old = hopping[bt]["n_element"]
                n_total += n_element_old + hopping_err.numel()
                ratio = n_element_old / (n_element_old + hopping_err.numel())

                hopping[bt] = {
                    "rmse": ((hopping[bt]["rmse"]**2) * ratio + (rmserr**2).mean() * (1-ratio)).sqrt(),
                    "mae":hopping[bt]["mae"] * ratio + maerr.mean() * (1-ratio),
                    "rmse_per_block_element": ((hopping[bt]["rmse_per_block_element"]**2) * ratio + rmserr**2 * (1-ratio)).sqrt(),
                    "mae_per_block_element": hopping[bt]["mae_per_block_element"]*ratio + maerr * (1-ratio),
                    "rmse_per_irreps": ((hopping[bt]["rmse_per_irreps"]**2) * ratio + rmse_per_irreps**2 * (1-ratio)).sqrt(),
                    "mae_per_irreps": hopping[bt]["mae_per_irreps"] * ratio + maerr_per_irreps * (1-ratio),
                    "n_element":n_element_old + hopping_err.numel(), 
                    }
                
                self.stats["mae"] += hopping[bt]["mae"] * hopping[bt]["n_element"]
                self.stats["rmse"] += hopping[bt]["rmse"]**2 * hopping[bt]["n_element"]
            
            if self.overlap:
                err = data[AtomicDataDict.EDGE_OVERLAP_KEY] - ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
                amp = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY].abs()
                mask = self.idp.mask_to_erme
                hopping = self.stats.get("overlap", {})

                for bt, tp in self.idp.bond_to_type.items():
                    hopping_mask = mask[tp]
                    hopping_err = err[data["edge_type"].flatten().eq(tp)]
                    if hopping_err.shape[0] == 0:
                        continue
                    hopping_err = hopping_err[:, hopping_mask]
                    
                    rmserr = (hopping_err**2).mean(dim=0).sqrt()
                    maerr = hopping_err.abs().mean(dim=0)
                    rmse_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                    rmse_per_irreps[hopping_mask] = rmserr
                    maerr_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                    maerr_per_irreps[hopping_mask] = maerr

                    rmse_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, rmse_per_irreps)
                    maerr_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, maerr_per_irreps)
                    

                    n_element_old = hopping[bt]["n_element"]
                    n_total += n_element_old + hopping_err.numel()
                    ratio = n_element_old / (n_element_old + hopping_err.numel())

                    hopping[bt] = {
                        "rmse": ((hopping[bt]["rmse"]**2) * ratio + (rmserr**2).mean() * (1-ratio)).sqrt(),
                        "mae":hopping[bt]["mae"] * ratio + maerr.mean() * (1-ratio),
                        "rmse_per_block_element": ((hopping[bt]["rmse_per_block_element"]**2) * ratio + rmserr**2 * (1-ratio)).sqrt(),
                        "mae_per_block_element": hopping[bt]["mae_per_block_element"]*ratio + maerr * (1-ratio),
                        "rmse_per_irreps": ((hopping[bt]["rmse_per_irreps"]**2) * ratio + rmse_per_irreps**2 * (1-ratio)).sqrt(),
                        "mae_per_irreps": hopping[bt]["mae_per_irreps"] * ratio + maerr_per_irreps * (1-ratio),
                        "n_element":n_element_old + hopping_err.numel(), 
                        }
                    
                    self.stats["mae"] += hopping[bt]["mae"] * hopping[bt]["n_element"]
                    self.stats["rmse"] += hopping[bt]["rmse"]**2 * hopping[bt]["n_element"]

            # compute overall mae, rmse
                    
            self.stats["mae"] = self.stats["mae"] / (n_total + 1e-6)
            self.stats["rmse"] = self.stats["rmse"] / (n_total + 1e-6)
            self.stats["rmse"] = self.stats["rmse"].sqrt()
            
        return self.stats
    
    def report(self):
        assert hasattr(self, "stats"), "The stats is not computed yet."

        print(f"TOTAL:")
        print(f"MAE: {self.stats['mae']}")
        print(f"RMSE: {self.stats['rmse']}")
        print(f"\n")
        
        with torch.no_grad():
            print(f"Onsite: ")
            for at, tp in self.idp.chemical_symbol_to_type.items():
                print(f"{at}:")
                print(f"MAE: {self.stats['onsite'][at]['mae']}")
                print(f"RMSE: {self.stats['onsite'][at]['rmse']}")

                # compute the onsite per block err
                onsite_mae = torch.zeros((self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
                onsite_rmse = torch.zeros((self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
                mae_per_block_element = torch.zeros((self.idp.reduced_matrix_element,), dtype=self.dtype, device=self.device)
                mae_per_block_element[_nrme_mask(self.idp, tp, result_device=mae_per_block_element.device)] = self.stats["onsite"][at]["mae_per_block_element"]
                rmse_per_block_element = torch.zeros((self.idp.reduced_matrix_element,), dtype=self.dtype, device=self.device)              
                rmse_per_block_element[_nrme_mask(self.idp, tp, result_device=rmse_per_block_element.device)] = self.stats["onsite"][at]["rmse_per_block_element"]
                
                ist = 0
                for i,iorb in enumerate(self.idp.full_basis):
                    jst = 0
                    li = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
                    for j,jorb in enumerate(self.idp.full_basis):
                        orbpair = iorb + "-" + jorb
                        lj = anglrMId[re.findall(r"[a-zA-Z]+", jorb)[0]]
                        
                        # constructing hopping blocks
                        if iorb == jorb:
                            factor = 0.5
                        else:
                            factor = 1.0

                        # constructing onsite blocks
                        if i <= j:
                            onsite_mae[ist:ist+2*li+1,jst:jst+2*lj+1] = factor * mae_per_block_element[self.idp.orbpair_maps[orbpair]].reshape(2*li+1, 2*lj+1)
                            onsite_rmse[ist:ist+2*li+1,jst:jst+2*lj+1] = factor * rmse_per_block_element[self.idp.orbpair_maps[orbpair]].reshape(2*li+1, 2*lj+1)

                        jst += 2*lj+1
                    ist += 2*li+1

                onsite_mae += onsite_mae.clone().T
                onsite_rmse += onsite_rmse.clone().T

                imask = _basis_mask(self.idp, tp, result_device=onsite_mae.device)
                onsite_mae = onsite_mae[imask][:,imask]
                onsite_rmse = onsite_rmse[imask][:,imask]

                vmax = onsite_mae.max().item()
                plt.matshow(onsite_mae.detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=vmax)
                plt.title("MAE")
                plt.colorbar()
                plt.show()

                vmax = onsite_rmse.max().item()
                plt.matshow(onsite_rmse.detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=vmax)
                plt.title("RMSE")
                plt.colorbar()
                plt.show()

            # compute the hopping per block err
            print(f"Hopping: ")
            for bt, tp in self.idp.bond_to_type.items():
                print(f"{bt}:")
                print(f"MAE: {self.stats['hopping'][bt]['mae']}")
                print(f"RMSE: {self.stats['hopping'][bt]['rmse']}")
                hopping_mae = torch.zeros((self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
                hopping_rmse = torch.zeros((self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
                mae_per_block_element = torch.zeros((self.idp.reduced_matrix_element,), dtype=self.dtype, device=self.device)
                mae_per_block_element[_erme_mask(self.idp, tp, result_device=mae_per_block_element.device)] = self.stats["hopping"][bt]["mae_per_block_element"]
                rmse_per_block_element = torch.zeros((self.idp.reduced_matrix_element,), dtype=self.dtype, device=self.device)              
                rmse_per_block_element[_erme_mask(self.idp, tp, result_device=rmse_per_block_element.device)] = self.stats["hopping"][bt]["rmse_per_block_element"]
                ist = 0
                for i,iorb in enumerate(self.idp.full_basis):
                    jst = 0
                    li = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
                    for j,jorb in enumerate(self.idp.full_basis):
                        orbpair = iorb + "-" + jorb
                        lj = anglrMId[re.findall(r"[a-zA-Z]+", jorb)[0]]
                        
                        # constructing hopping blocks
                        if iorb == jorb:
                            factor = 0.5
                        else:
                            factor = 1.0

                        # constructing onsite blocks
                        if i <= j:
                            hopping_mae[ist:ist+2*li+1,jst:jst+2*lj+1] = factor * mae_per_block_element[self.idp.orbpair_maps[orbpair]].reshape(2*li+1, 2*lj+1)
                            hopping_rmse[ist:ist+2*li+1,jst:jst+2*lj+1] = factor * rmse_per_block_element[self.idp.orbpair_maps[orbpair]].reshape(2*li+1, 2*lj+1)

                        jst += 2*lj+1
                    ist += 2*li+1

                hopping_mae += hopping_mae.clone().T
                hopping_rmse += hopping_rmse.clone().T
                
                iat, jat = bt.split("-")
                imask = _basis_mask(self.idp, self.idp.chemical_symbol_to_type[iat], result_device=hopping_mae.device)
                jmask = _basis_mask(self.idp, self.idp.chemical_symbol_to_type[jat], result_device=hopping_mae.device)
                hopping_mae = hopping_mae[imask][:,jmask]
                hopping_rmse = hopping_rmse[imask][:,jmask]

                vmax = hopping_mae.max().item()
                plt.matshow(hopping_mae.detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=vmax)
                plt.title("MAE")
                plt.colorbar()
                plt.show()

                vmax = hopping_mae.max().item()
                plt.matshow(hopping_rmse.detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=vmax)
                plt.title("RMSE")
                plt.colorbar()
                plt.show()

        



    def __cal_norm__(self, irreps: Irreps, x: torch.Tensor):
        id = 0
        out = []
        if len(x.shape) == 1:
            x = x.unsqueeze_(0)
        for mul, ir in irreps:
            tensor = x[:,id:id+mul*ir.dim].reshape(-1, mul, ir.dim)
            id = id + mul*ir.dim
            tensor = tensor.norm(dim=-1)
            out.append(tensor)

        return torch.cat(out, dim=-1).squeeze(0)


try:
    from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss  # noqa: F401
except Exception as _blockwise_exc:  # pragma: no cover
    log.warning("Could not register hamil_blockwise_nextham: %s", _blockwise_exc)


def _unnest(t):
    """Unwrap a single-graph nested tensor; pass anything else through."""
    if torch.is_tensor(t) and t.is_nested:
        return t[0]
    return t


@Loss.register("eig_ham_h0res")
class EigHamH0ResLoss(nn.Module):
    """Joint H-matrix + band loss for an arm whose target is an H0 residual.

    ``eig_ham`` assumes NODE/EDGE_FEATURES already hold the physical
    Hamiltonian. The h0dh arm predicts dH against a stored H0 prior, so the
    band term has to diagonalize H0 + dH while the matrix term keeps scoring
    the residual exactly as pretraining did -- otherwise the two halves of the
    objective disagree about what the model outputs.

    total = coeff_ham * L_ham(residual) + (1 - coeff_ham) * L_eig(H0 + residual)

    Keeping L_ham in the mix is what makes 138 band labels survivable: it
    anchors the model to the matrix it already fits, so the band term only
    steers rather than redefines the target.
    """

    def __init__(
        self,
        basis: Dict[str, Union[str, list]] = None,
        idp: Union[OrbitalMapper, None] = None,
        band_overlap: bool = True,
        coeff_ham: float = 0.9,
        band_emin: float = None,
        band_emax: float = None,
        band_min: int = 0,
        band_max: int = None,
        eout_weight: float = 0.01,
        diff_on: bool = False,
        diff_weight: float = 0.01,
        diff_valence: dict = None,
        spin_deg: int = 2,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        super(EigHamH0ResLoss, self).__init__()
        if not 0.0 <= coeff_ham <= 1.0:
            raise ValueError(f"coeff_ham must be in [0, 1], got {coeff_ham}.")
        self.coeff_ham = float(coeff_ham)
        self.device = device

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        # Same matrix loss the pretraining run used, so the anchor term is
        # numerically comparable to the checkpoint's own train_loss.
        # Trainer merges common_options into every loss kwargs, so basis /
        # overlap / dtype / device arrive twice; drop the copies passed
        # explicitly. overlap stays False: this model predicts no S, and the
        # matrix term must score exactly what pretraining scored.
        ham_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ("basis", "overlap", "dtype", "device", "idp")}
        self.ham_loss = HamilLossAbs(
            idp=self.idp, overlap=False, dtype=dtype, device=device, **ham_kwargs
        )
        # Direct eigensolver rather than EigLoss: see module note on the
        # Batch.from_dict/to_data_list mis-slice.
        self.eigen = Eigenvalues(
            idp=self.idp,
            h_edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            h_node_field=AtomicDataDict.NODE_FEATURES_KEY,
            h_out_field=AtomicDataDict.HAMILTONIAN_KEY,
            out_field=AtomicDataDict.ENERGY_EIGENVALUE_KEY,
            s_edge_field=AtomicDataDict.EDGE_OVERLAP_KEY if band_overlap else None,
            s_node_field=AtomicDataDict.NODE_OVERLAP_KEY if band_overlap else None,
            s_out_field=AtomicDataDict.OVERLAP_KEY if band_overlap else None,
            dtype=dtype,
            device=device,
        )
        self.eout_weight = eout_weight

        self.band_emin = band_emin
        self.band_emax = band_emax
        self.band_min = band_min
        self.band_max = band_max
        self._last_parts = {}

    def _add_h0(self, src: AtomicDataDict) -> dict:
        """Shallow copy with the physical Hamiltonian in the feature fields."""
        for key in (AtomicDataDict.NODE_H0_KEY, AtomicDataDict.EDGE_H0_KEY):
            if src.get(key, None) is None:
                raise KeyError(
                    f"eig_ham_h0res needs {key!r} to rebuild the physical "
                    "Hamiltonian; run the dataset with get_H0=true."
                )
        out = dict(src)
        out[AtomicDataDict.NODE_FEATURES_KEY] = (
            src[AtomicDataDict.NODE_FEATURES_KEY] + src[AtomicDataDict.NODE_H0_KEY]
        )
        out[AtomicDataDict.EDGE_FEATURES_KEY] = (
            src[AtomicDataDict.EDGE_FEATURES_KEY] + src[AtomicDataDict.EDGE_H0_KEY]
        )
        return out



    # Only these reach the eigensolver. A collated batch also carries nested
    # k-point/eigenvalue tensors and batch bookkeeping, which HR2HK cannot size.
    _SOLVER_FIELDS = (
        AtomicDataDict.EDGE_INDEX_KEY,
        AtomicDataDict.EDGE_CELL_SHIFT_KEY,
        AtomicDataDict.POSITIONS_KEY,
        AtomicDataDict.CELL_KEY,
        AtomicDataDict.PBC_KEY,
        AtomicDataDict.ATOM_TYPE_KEY,
        AtomicDataDict.EDGE_TYPE_KEY,
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
        AtomicDataDict.NODE_FEATURES_KEY,
        AtomicDataDict.EDGE_FEATURES_KEY,
    )

    def _solver_dict(self, src: dict, ref: dict) -> dict:
        """Assemble the eigensolver input.

        Hamiltonian RMEs come from ``src`` (the model output); the overlap and
        the k-points come from ``ref`` (the untouched batch). The model
        overwrites EDGE_OVERLAP_KEY with internal 128-dim latents during
        forward, so reading S from the prediction gives garbage of the wrong
        width.
        """
        out = {}
        for key in self._SOLVER_FIELDS:
            value = src.get(key, None)
            if value is not None:
                out[key] = _unnest(value)

        n_rme = self.idp.reduced_matrix_element
        for key in (AtomicDataDict.NODE_OVERLAP_KEY,
                    AtomicDataDict.EDGE_OVERLAP_KEY):
            value = _unnest(ref.get(key, None))
            if value is None:
                raise KeyError(
                    f"band term needs {key!r} on the reference batch; run the "
                    "dataset with get_overlap=true."
                )
            if value.shape[-1] != n_rme:
                raise ValueError(
                    f"{key} has width {value.shape[-1]}, expected the RME width "
                    f"{n_rme}. A width of 128 means the model overwrote this "
                    "field with internal latents -- read S from the reference "
                    "batch, not the prediction."
                )
            out[key] = value

        kpoint = _unnest(ref.get(AtomicDataDict.KPOINT_KEY, None))
        if kpoint is None:
            kpoint = _unnest(src.get(AtomicDataDict.KPOINT_KEY, None))
        if kpoint is None:
            raise KeyError("band term needs k-points on the batch.")
        out[AtomicDataDict.KPOINT_KEY] = kpoint.reshape(-1, 3)
        return out

    def _band_loss(self, pred_phys: dict, ref_phys: dict):
        """Windowed MSE between predicted and reference bands, one graph.

        Follows EigLoss's conventions so the number stays comparable: each side
        is shifted by its own minimum, the window is measured in that
        bottom-relative coordinate, and bands outside it are down-weighted to
        eout_weight rather than dropped.
        """
        n_graph = 1
        ptr = pred_phys.get(AtomicDataDict.BATCH_PTR_KEY, None)
        if ptr is not None:
            n_graph = int(ptr.numel()) - 1
        if n_graph != 1:
            raise RuntimeError(
                f"eig_ham_h0res band term expects one graph per batch, got "
                f"{n_graph}. k-points and band counts are ragged across "
                f"structures, so set batch_size=1."
            )

        solver_input = self._solver_dict(pred_phys, ref_phys)
        out = self.eigen(solver_input)
        eig_pred = out[AtomicDataDict.ENERGY_EIGENVALUE_KEY]
        if eig_pred.dim() == 3:
            eig_pred = eig_pred[0]
        eig_ref = _unnest(ref_phys[AtomicDataDict.ENERGY_EIGENVALUE_KEY])
        if torch.is_tensor(eig_ref) and eig_ref.dim() == 3:
            eig_ref = eig_ref[0]
        eig_ref = eig_ref.to(device=eig_pred.device, dtype=eig_pred.dtype)

        if eig_pred.shape[0] != eig_ref.shape[0]:
            raise RuntimeError(
                f"k-point count differs: pred {eig_pred.shape[0]} vs ref "
                f"{eig_ref.shape[0]}."
            )
        nb = min(eig_pred.shape[1], eig_ref.shape[1])
        lo = int(self.band_min)
        hi = int(self.band_max) if self.band_max is not None else nb
        hi = min(hi, nb)
        if lo >= hi:
            raise RuntimeError(f"empty band window: band_min={lo} band_max={hi}.")

        p = eig_pred[:, lo:hi]
        r = eig_ref[:, lo:hi]
        p = p - p.reshape(-1).min()
        r = r - r.reshape(-1).min()

        diff2 = (p - r) ** 2
        if self.band_emin is None and self.band_emax is None:
            return diff2.mean()

        mask = torch.ones_like(r, dtype=torch.bool)
        if self.band_emin is not None:
            mask &= r > self.band_emin
        if self.band_emax is not None:
            mask &= r < self.band_emax
        n_in = int(mask.sum())
        n_out = mask.numel() - n_in
        loss = diff2.new_zeros(())
        if n_in:
            loss = loss + diff2[mask].mean()
        if n_out:
            loss = loss + self.eout_weight * diff2[~mask].mean()
        return loss

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        ham_loss = self.ham_loss(data, ref_data)

        if self.coeff_ham >= 1.0:
            self._last_parts = {
                "ham": float(ham_loss.detach()), "eig": 0.0}
            return ham_loss

        pred_phys = self._add_h0(data)
        ref_phys = self._add_h0(ref_data)

        # The window lives on the loss, not in the dataset: it is a training
        # knob, and putting it here keeps the LMDB records reusable.
        if self.band_emin is not None or self.band_emax is not None:
            ref_phys[AtomicDataDict.ENERGY_WINDOWS_KEY] = (
                self.band_emin, self.band_emax)
        if self.band_max is not None:
            ref_phys[AtomicDataDict.BAND_WINDOW_KEY] = (
                self.band_min, self.band_max)

        eig_loss = self._band_loss(pred_phys, ref_phys)
        self._last_parts = {
            "ham": float(ham_loss.detach()), "eig": float(eig_loss.detach())}
        return self.coeff_ham * ham_loss + (1.0 - self.coeff_ham) * eig_loss


@Loss.register("hamil_abs_gauged")
class HamilAbsGaugedLoss(nn.Module):
    """hamil_abs, with the H -> H + mu*S gauge freedom removed first.

    Arm A of the NextHAM port: no k-space term, no eigendecomposition, no band
    labels. Only the overlap is needed beyond what pretraining used.
    """

    def __init__(
        self,
        basis: Dict[str, Union[str, list]] = None,
        idp: Union[OrbitalMapper, None] = None,
        gauge: bool = True,
        gauge_clip: float = 1.0,
        **kwargs,
    ):
        super(HamilAbsGaugedLoss, self).__init__()
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb",
                                     device=kwargs.get("device", torch.device("cpu")))
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        inner = {k: v for k, v in kwargs.items() if k not in ("basis", "overlap", "idp")}
        self.ham_loss = HamilLossAbs(idp=self.idp, overlap=False, **inner)
        self.gauge = bool(gauge)
        self.gauge_clip = float(gauge_clip)
        self._last_parts = {}

    def _solve_mu(self, data, ref_data):
        """Closed-form mu on the real-space projection, detached.

        mu = <dH_pred - dH_ref, S> / <S, S>, summed over the valid RME entries
        of both node and edge blocks. Detached so no gradient flows through the
        gauge solve itself.
        """
        num = den = 0.0
        for feat_key, ovp_key in (
            (AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
            (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY),
        ):
            s = ref_data.get(ovp_key, None)
            if s is None:
                raise KeyError(
                    f"hamil_abs_gauged needs {ovp_key!r} on the reference batch; "
                    "run the dataset with get_overlap=true."
                )
            if torch.is_tensor(s) and s.is_nested:
                s = s[0]
            n_rme = self.idp.reduced_matrix_element
            if s.shape[-1] != n_rme:
                raise ValueError(
                    f"{ovp_key} has width {s.shape[-1]}, expected {n_rme}. "
                    "A width of 128 means the model overwrote this field -- "
                    "read S from the reference batch, not the prediction."
                )
            d = (data[feat_key].detach() - ref_data[feat_key].detach())
            num = num + (d * s).sum()
            den = den + (s * s).sum()
        mu = num / den.clamp_min(1e-30)
        return float(mu.clamp(-self.gauge_clip, self.gauge_clip))

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        if not self.gauge:
            loss = self.ham_loss(data, ref_data)
            self._last_parts = {"mu": 0.0, "ham": float(loss.detach()),
                                "ham_ungauged": float(loss.detach())}
            return loss

        with torch.no_grad():
            ungauged = float(self.ham_loss(data, ref_data).detach())

        mu = self._solve_mu(data, ref_data)

        # Shift the TARGET, not the prediction: the gauge belongs to the label.
        shifted = dict(ref_data)
        for feat_key, ovp_key in (
            (AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
            (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY),
        ):
            s = ref_data[ovp_key]
            if torch.is_tensor(s) and s.is_nested:
                s = s[0]
            shifted[feat_key] = ref_data[feat_key] + mu * s

        loss = self.ham_loss(data, shifted)
        self._last_parts = {"mu": mu, "ham": float(loss.detach()),
                            "ham_ungauged": ungauged,
                            "gauge_gain": ungauged / max(float(loss.detach()), 1e-30)}
        return loss


@Loss.register("nextham_kspace")
class NextHAMKSpaceLoss(nn.Module):
    """Real-space H loss plus NextHAM's P/Q/PQ projection loss."""

    def __init__(
        self,
        basis: Dict[str, Union[str, list]] = None,
        idp: Union[OrbitalMapper, None] = None,
        w_p: float = 2e-4,
        w_q: float = 1e-4,
        w_pq: float = 1.5e-4,
        gauge: bool = True,
        gauge_clip: float = 1.0,
        band_window: float = 10.0,
        q_window: float = 30.0,
        n_kpoints: int = 1,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        super(NextHAMKSpaceLoss, self).__init__()
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        inner = {k: v for k, v in kwargs.items()
                 if k not in ("basis", "overlap", "idp", "dtype", "device")}
        self.ham_loss = HamilLossAbs(idp=self.idp, overlap=False,
                                     dtype=dtype, device=device, **inner)
        self.w_p, self.w_q, self.w_pq = float(w_p), float(w_q), float(w_pq)
        self.w_r = 1.0 - (self.w_p + self.w_q + self.w_pq)
        if self.w_r <= 0:
            raise ValueError("k-space weights sum to >= 1; nothing left for H(R).")
        self.gauge = bool(gauge)
        self.gauge_clip = float(gauge_clip)
        self.band_window = band_window
        self.q_window = q_window
        self.n_kpoints = int(n_kpoints)
        self.device = device
        self._dtype = dtype
        self.l1 = nn.L1Loss(reduction="mean")
        self._last_parts = {}

        self.h2k = HR2HK(idp=self.idp, edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                         node_field=AtomicDataDict.NODE_FEATURES_KEY,
                         out_field=AtomicDataDict.HAMILTONIAN_KEY,
                         dtype=dtype, device=device)
        self.s2k = HR2HK(idp=self.idp, overlap=True,
                         edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                         node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                         out_field=AtomicDataDict.OVERLAP_KEY,
                         dtype=dtype, device=device)

    # ---- gauge -----------------------------------------------------------
    def _solve_mu(self, data, ref_data):
        num = den = 0.0
        for fk, ok in ((AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
                       (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY)):
            s = _unnest(ref_data.get(ok, None))
            if s is None:
                raise KeyError("nextham_kspace needs %r; run with get_overlap=true." % ok)
            if s.shape[-1] != self.idp.reduced_matrix_element:
                raise ValueError(
                    "%s width %d != RME width %d (128 means the model overwrote it)"
                    % (ok, s.shape[-1], self.idp.reduced_matrix_element))
            d = data[fk].detach() - ref_data[fk].detach()
            num = num + (d * s).sum()
            den = den + (s * s).sum()
        return float((num / den.clamp_min(1e-30)).clamp(-self.gauge_clip, self.gauge_clip))

    # ---- k-space ---------------------------------------------------------
    def _base_fields(self, ref_data):
        keys = (AtomicDataDict.EDGE_INDEX_KEY, AtomicDataDict.EDGE_CELL_SHIFT_KEY,
                AtomicDataDict.POSITIONS_KEY, AtomicDataDict.CELL_KEY,
                AtomicDataDict.PBC_KEY, AtomicDataDict.ATOM_TYPE_KEY,
                AtomicDataDict.EDGE_TYPE_KEY, AtomicDataDict.ATOMIC_NUMBERS_KEY)
        return {k: _unnest(ref_data[k]) for k in keys if k in ref_data}

    def _split_pq(self, eigvals, nelec):
        """P/Q by energy window around E_F, estimated from the electron count."""
        n_occ = max(1, int(math.ceil(float(nelec) / 2.0)))
        n_occ = min(n_occ, eigvals.numel() - 1)
        e_f = float(eigvals[n_occ - 1])
        rel = eigvals - e_f
        p_mask = (rel >= -self.band_window) & (rel <= self.band_window)
        if self.q_window is None:
            q_mask = rel > self.band_window
        else:
            q_mask = (rel > self.band_window) & (rel <= self.band_window + self.q_window)
        return p_mask, q_mask

    def _kspace_terms(self, data, ref_data, mu):
        self._empty_pq = 0
        for key in (AtomicDataDict.NODE_H0_KEY, AtomicDataDict.EDGE_H0_KEY):
            if ref_data.get(key, None) is None:
                raise KeyError("nextham_kspace needs %r to rebuild the physical "
                               "Hamiltonian; run the dataset with get_H0=true." % key)
        base = self._base_fields(ref_data)
        n_orb_probe = None
        acc = {"p": [], "q": [], "pq": []}

        for _ in range(self.n_kpoints):
            kpt = torch.rand(1, 3, device=data[AtomicDataDict.EDGE_FEATURES_KEY].device,
                             dtype=torch.get_default_dtype())
            d_lab = dict(base)
            d_lab[AtomicDataDict.KPOINT_KEY] = kpt
            # The eigenbasis must come from the physical H = H0 + dH. H0 cancels
            # in the difference terms but not inside the diagonalisation.
            nh0 = _unnest(ref_data[AtomicDataDict.NODE_H0_KEY])
            eh0 = _unnest(ref_data[AtomicDataDict.EDGE_H0_KEY])
            d_lab[AtomicDataDict.NODE_FEATURES_KEY] = _unnest(ref_data[AtomicDataDict.NODE_FEATURES_KEY]) + nh0
            d_lab[AtomicDataDict.EDGE_FEATURES_KEY] = _unnest(ref_data[AtomicDataDict.EDGE_FEATURES_KEY]) + eh0
            d_lab[AtomicDataDict.NODE_OVERLAP_KEY] = _unnest(ref_data[AtomicDataDict.NODE_OVERLAP_KEY])
            d_lab[AtomicDataDict.EDGE_OVERLAP_KEY] = _unnest(ref_data[AtomicDataDict.EDGE_OVERLAP_KEY])

            with torch.no_grad():
                sk = self.s2k(dict(d_lab))[AtomicDataDict.OVERLAP_KEY][0]
                hk_lab = self.h2k(dict(d_lab))[AtomicDataDict.HAMILTONIAN_KEY][0]
                # Generalised eigenproblem in the label's own gauge.
                lo = torch.linalg.cholesky(sk)
                lo_inv = torch.linalg.inv(lo)
                heff = lo_inv @ hk_lab @ lo_inv.conj().transpose(-1, -2)
                evals, evecs = torch.linalg.eigh(heff)
                u = lo_inv.conj().transpose(-1, -2) @ evecs      # columns: eigenvectors
                nelec = float(ref_data.get("nelec", evals.numel()))
                p_mask, q_mask = self._split_pq(evals.real, nelec)
                u_p = u[:, p_mask]
                u_q = u[:, q_mask]
            if u_p.shape[1] == 0 or u_q.shape[1] == 0:
                # Not a silent skip: a spectrum that cannot fill both windows
                # means the split (or the Hamiltonian) is wrong.
                self._empty_pq += 1
                continue
            n_orb_probe = sk.shape[0]

            d_pred = dict(d_lab)
            d_pred[AtomicDataDict.NODE_FEATURES_KEY] = data[AtomicDataDict.NODE_FEATURES_KEY] + nh0
            d_pred[AtomicDataDict.EDGE_FEATURES_KEY] = data[AtomicDataDict.EDGE_FEATURES_KEY] + eh0
            hk_pred = self.h2k(d_pred)[AtomicDataDict.HAMILTONIAN_KEY][0]

            eye_p = torch.eye(u_p.shape[1], device=hk_pred.device, dtype=hk_pred.dtype)
            eye_q = torch.eye(u_q.shape[1], device=hk_pred.device, dtype=hk_pred.dtype)
            for tag, ua, ub, eye in (("p", u_p, u_p, eye_p),
                                     ("q", u_q, u_q, eye_q),
                                     ("pq", u_p, u_q, None)):
                a = ua.conj().transpose(-1, -2) @ hk_lab @ ub
                b = ua.conj().transpose(-1, -2) @ hk_pred @ ub
                if eye is not None:
                    a = a + mu * eye
                acc[tag].append(self.l1(a.real, b.real) + self.l1(a.imag, b.imag))

        def red(v):
            return torch.stack(v).mean() if v else torch.zeros((), device=self.device)
        return red(acc["p"]), red(acc["q"]), red(acc["pq"]), n_orb_probe

    # ---- forward ---------------------------------------------------------
    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        mu = self._solve_mu(data, ref_data) if self.gauge else 0.0

        shifted = dict(ref_data)
        if mu:
            for fk, ok in ((AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
                           (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY)):
                shifted[fk] = ref_data[fk] + mu * _unnest(ref_data[ok])
        l_r = self.ham_loss(data, shifted)

        l_p, l_q, l_pq, n_orb = self._kspace_terms(data, ref_data, mu)
        total = self.w_r * l_r + self.w_p * l_p + self.w_q * l_q + self.w_pq * l_pq

        self._last_parts = {
            "mu": mu, "n_orb": n_orb, "empty_pq": getattr(self, "_empty_pq", 0),
            "R": float(l_r.detach()), "P": float(l_p.detach()),
            "Q": float(l_q.detach()), "PQ": float(l_pq.detach()),
            "wR": self.w_r * float(l_r.detach()),
            "wP": self.w_p * float(l_p.detach()),
            "wQ": self.w_q * float(l_q.detach()),
            "wPQ": self.w_pq * float(l_pq.detach()),
        }
        return total
