"""Explicit inference-only fused plans; the table bank default is unchanged."""
import struct
from functools import lru_cache
import torch
from ._cuda import extension,check_device


@lru_cache(maxsize=512)
def _same_angular(first, table, pointers):
    return all(torch.equal(getattr(first,n),getattr(table,n)) for n in
               ('cuda_degrees','cuda_directions','cuda_inverse','cuda_scales'))


class RadialMultiPlan:
    """One launch for heterogeneous groups of co-rotated 1-D tables.

    Each group is (tables, query_count). Tables in a group must have identical
    angular metadata. A plan owns strong references to its immutable buffers;
    rebuild after moving/replacing tables. ``background_nodes`` reserves the
    2-D background-axis contract and rejects it until implemented.
    """
    def __init__(self, groups, *, background_nodes=None):
        if background_nodes is not None:raise NotImplementedError('2-D background interpolation is not implemented')
        if not groups:raise ValueError('at least one table group is required')
        self.tables=[];self.views=[];descs=[];group_rows=[];query_groups=[];qoffset=offset=0;self.max_rotation=0
        anchor=groups[0][0][0].knots
        if not anchor.is_cuda:raise ValueError('multi radial requires CUDA tables')
        self.device,self.dtype=anchor.device,anchor.dtype
        self.pointers=[]
        angular=('cuda_degrees','cuda_directions','cuda_inverse','cuda_scales')
        buffers=('knots','coefficients',*angular,'cuda_ptr','cuda_terms','cuda_canonical')
        for gid,(tables,count) in enumerate(groups):
            if not tables or not isinstance(count,int) or count<0:raise ValueError('invalid radial group')
            first=tables[0];group_rows.append([qoffset,len(descs),len(tables),count]);query_groups.extend([gid]*count)
            # This check is preparation-only, never part of the kernel launch.
            for table in tables:
                if table is not first and not _same_angular(first,table,tuple(getattr(t,n).data_ptr() for t in (first,table) for n in angular)):
                    raise ValueError('co-rotation requires identical angular metadata')
                vals=[getattr(table,name) for name in buffers]
                if any(t.device!=self.device or not t.is_contiguous() for t in vals) or table.knots.dtype!=self.dtype:raise ValueError('tables must share contiguous device/dtype buffers')
                ptrs=[v.data_ptr() for v in vals];self.pointers.append(ptrs)
                width=table.cuda_canonical.numel()
                descs.append(ptrs+[table.knots.numel(),table.coefficients.shape[2],len(table.cuda_degrees),table.cuda_inverse.numel(),width,struct.unpack('q',struct.pack('d',table.support_bohr))[0],offset])
                self.views.append((offset,count,table.shape));offset+=count*width
                self.tables.append(table);self.max_rotation=max(self.max_rotation,table.cuda_inverse.numel())
            qoffset+=count
        self.size=offset;self.nqueries=qoffset;self.buffers=buffers
        self.query_groups=torch.tensor(query_groups,device=self.device,dtype=torch.long)
        self.groups=torch.tensor(group_rows,device=self.device,dtype=torch.long)
        self.descriptors=torch.tensor(descs,device=self.device,dtype=torch.long)

    def __call__(self,vectors):
        if vectors.shape!=(self.nqueries,3) or vectors.device!=self.device or vectors.dtype!=self.dtype or vectors.requires_grad:
            raise ValueError('multi vectors must match the inference plan')
        for table,pointers in zip(self.tables,self.pointers):
            if [getattr(table,n).data_ptr() for n in self.buffers]!=pointers:raise RuntimeError('table buffers moved; rebuild multi plan')
        check_device(vectors.device)
        out=extension().radial_multi(vectors.contiguous(),self.query_groups,self.groups,self.descriptors,self.size,self.max_rotation)
        return [out[offset:offset+count*shape[0]*shape[1]].view(count,*shape) for offset,count,shape in self.views]


def contract_add(a,m,b,rows,target):
    """FP64/FP32 gather(A)^T M gather(B), atomically accumulated in target.

    M is a diagonal vector or a dense real matrix. Autograd and complex SOC
    use the existing Torch path. Reduction ordering is not bitwise stable.
    """
    if any(x.requires_grad for x in (a,m,b,target)):raise ValueError('contraction is inference-only')
    check_device(a.device)
    extension().contract_add(a,m,b,rows,target)


def enable_fusion(plan, *, radial=True, contraction=False):
    """Opt in after prepare(); contraction stays explicit because FP32 packing
    can magnify tiny FP64 reduction changes across a rounding midpoint.
    """
    if plan.positions.device.type!='cuda':raise ValueError('fusion requires a CUDA plan')
    plan.fused_contraction=contraction
    if contraction and hasattr(plan,'base_specs'):
        for number,kind,*_ in plan.contraction_specs:
            if kind=='vna':
                plan.register_buffer(f'fused_matrix_{number}',getattr(plan,f'matrix_{number}').diagonal().contiguous())
    if not radial:return plan
    groups=[];specs=[];used=set()
    assembly=len(plan.query_specs[0])==3 if plan.query_specs else hasattr(plan,'base_specs')
    for spec in plan.query_specs:
        number=spec[0]
        if number in used:continue
        ids=[number];key=spec[-1]
        if assembly and spec[1][0]=='p2_base':
            other=plan.pair_ids.get(('overlap',*spec[1][1:]))
            if other is not None and other not in used and torch.equal(getattr(plan,f'query_{number}'),getattr(plan,f'query_{other}')):ids.append(other)
        lookup={s[0]:s[-1] for s in plan.query_specs}
        tables=[plan.bank.tables[lookup[i]] for i in ids]
        groups.append((tables,len(getattr(plan,f'query_{number}'))));specs.append((number,ids));used.update(ids)
    plan.fused_radial_specs=specs
    plan.fused_radial=RadialMultiPlan(groups) if groups else None
    return plan


def evaluate_plan(plan):
    vectors=[]
    for number,ids in plan.fused_radial_specs:
        q=getattr(plan,f'query_{number}')
        cell=plan.cell if hasattr(plan,'cell') else plan.cells
        translation=(torch.einsum('ei,eij->ej',q[:,2:5].to(cell.dtype),cell[q[:,5]]) if cell.ndim==3 else q[:,2:5].to(cell.dtype)@cell)
        vectors.append(plan.positions[q[:,0]]-plan.positions[q[:,1]]+translation)
    if not vectors:return {}
    outputs=plan.fused_radial(torch.cat(vectors));values={};index=0
    kinds={s[0]:s[1][0] for s in plan.query_specs} if hasattr(plan,'base_specs') else {}
    for delta,(number,ids) in zip(vectors,plan.fused_radial_specs):
        for i in ids:
            block=outputs[index];index+=1
            if kinds.get(i)=='projector':
                active=torch.linalg.vector_norm(delta,dim=-1)[:,None]<=getattr(plan,f'cutoffs_{i}')[None,:]
                block=block*active[:,:,None]
            values[i]=block
    return values
