import numpy as np
import torch
from typing import List
from dptb.structure.abstract_stracture import AbstractStructure

class Processor(object):
    def __init__(self, structure_list: List[AbstractStructure], kpoint, eigen_list, batchsize: int, env_cutoff: float, device='cpu', require_dict=False, dtype=torch.float32):
        super(Processor, self).__init__()
        if isinstance(structure_list, AbstractStructure):
            structure_list = [structure_list]
        self.structure_list = np.array(structure_list, dtype=object)
        self.kpoint = kpoint
        self.eigen_list= np.array(eigen_list, dtype=object)
        self.require_dict = require_dict

        self.n_st = len(self.structure_list)
        self.__struct_idx_unsampled__ = np.random.choice(np.array(list(range(len(self.structure_list)))),
                                                     size=len(self.structure_list), replace=False)

        self.__struct_unsampled__ = self.structure_list[self.__struct_idx_unsampled__]
        self.__struct_workspace__ = []
        self.__struct_idx_workspace__ = []
        self.env_cutoff = env_cutoff
        self.batchsize = batchsize
        self.n_batch = int(self.n_st / batchsize)
        if self.n_st % batchsize:
            self.n_batch += 1

        assert self.batchsize > 0

        self.device = device
        self.dtype = dtype

    def shuffle(self):
        '''> If the batch size is larger than the number of unsampled structures, then we sample all the
        remaining structures and reset the unsampled list to the full list of structures. Otherwise, we
        sample the first `batch_size` structures from the unsampled list and remove them from the unsampled list
        
        Parameters
        ----------
        batch_size int : number of structures to be sampled        
        '''

        if self.batchsize >= len(self.__struct_unsampled__):
            self.__struct_workspace__ = self.__struct_unsampled__
            self.__struct_idx_workspace__ = self.__struct_idx_unsampled__
            self.__struct_idx_unsampled__ = np.random.choice(np.array(list(range(len(self.structure_list)))),
                                                         size=len(self.structure_list), replace=False)
            self.__struct_unsampled__ = self.structure_list[self.__struct_idx_unsampled__]
        else:
            self.__struct_workspace__ = self.__struct_unsampled__[:self.batchsize]
            self.__struct_idx_workspace__ = self.__struct_idx_unsampled__[:self.batchsize]
            self.__struct_idx_unsampled__ = self.__struct_idx_unsampled__[self.batchsize:]
            self.__struct_unsampled__ = self.__struct_unsampled__[self.batchsize:]
    
    def get_env(self):
        '''It takes the environment of each structure in the workspace and concatenates them into one big
        environment
        
        Returns
        -------
            A dictionary of the environment for ent type for all the strucutes in  the works sapce.
        '''
        if len(self.__struct_workspace__) == 0:
            self.__struct_workspace__ = self.structure_list

        batch_env = {}
        n_stw = len(self.__struct_workspace__)
        for st in range(n_stw):
            if not self.__struct_workspace__[st].if_env_ready:
                self.__struct_workspace__[st].cal_env(env_cutoff=self.env_cutoff)
            env = self.__struct_workspace__[st].get_env()
            # (i,itype,s(r),rx,ry,rz)
            for ek in env.keys():
                # to envalue the order for each structure of the envs.
                env_ek = np.concatenate([np.ones((env[ek].shape[0], 1))*st, env[ek]], axis=1)

                if batch_env.get(ek) is None:
                    batch_env[ek] = env_ek
                else:
                    batch_env[ek] = np.concatenate([batch_env[ek], env_ek], axis=0)

        for ek in batch_env:
            batch_env[ek] = torch.tensor(batch_env[ek], dtype=self.dtype, device=self.device)

        return batch_env # {env_type: [f,i,itype,s(r),rx,ry,rz]}

    def get_bond(self):
        '''It takes the bonds of each structure in the workspace and concatenates them into one big dictionary.
        
        Returns
        -------
            A Tensor of the bonds lists for bond type for all the strucutes in the works space.
        '''

        if len(self.__struct_workspace__) == 0:
            self.__struct_workspace__ = self.structure_list

        if self.require_dict:
            batch_bond = {}
            batch_bond_onsite = {}
            n_stw = len(self.__struct_workspace__)
            for st in range(n_stw):
                bond, bond_onsite = self.__struct_workspace__[st].cal_bond()
                bond = np.concatenate([np.ones((bond.shape[0], 1)) * st, bond], axis=1)
                bond_onsite = np.concatenate([np.ones((bond_onsite.shape[0], 1)) * st, bond_onsite], axis=1)
                batch_bond.update({st:torch.tensor(bond, dtype=self.dtype, device=self.device)})
                batch_bond_onsite.update({st:torch.tensor(bond_onsite, dtype=self.dtype, device=self.device)})

        else:
            batch_bond = []
            batch_bond_onsite = []
            n_stw = len(self.__struct_workspace__)
            for st in range(n_stw):
                bond, bond_onsite = self.__struct_workspace__[st].cal_bond()
                bond = np.concatenate([np.ones((bond.shape[0], 1)) * st, bond], axis=1)
                bond_onsite = np.concatenate([np.ones((bond_onsite.shape[0], 1)) * st, bond_onsite], axis=1)
                batch_bond.append(torch.tensor(bond, dtype=self.dtype, device=self.device))
                batch_bond_onsite.append(torch.tensor(bond_onsite, dtype=self.dtype, device=self.device))

            batch_bond = torch.cat(batch_bond, dim=0)
            batch_bond_onsite = torch.cat(batch_bond_onsite, dim=0)

        return batch_bond, batch_bond_onsite # [f, i_atom_num, i, j_atom_num, j, Rx, Ry, Rz, |rj-ri|, \hat{rij: x, y, z}] or dict

    @property
    def atom_type(self):
        '''It returns a list of unique atom types in the structure.
        
        Returns
        -------
            A list of unique atom types.
        '''
        at_list = []
        for st in self.structure_list:
            at_list += st.atom_type

        return list(set(at_list))

    @property
    def proj_atom_type(self):
        ''' This function returns a list of all the projected atom types in the structure list
        
        Returns
        -------
            A list of unique atom types in the structure list.
        '''
        at_list = []
        for st in self.structure_list:
            at_list += st.proj_atom_type

        return list(set(at_list))

    def __iter__(self):
        # processor = Processor; for i in processor: i: (batch_bond, batch_env, structures)
        self.it = 0 # label of iteration
        self.__struct_idx_unsampled__ = np.random.choice(np.array(list(range(len(self.structure_list)))),
                                                         size=len(self.structure_list), replace=False)

        self.__struct_unsampled__ = self.structure_list[self.__struct_idx_unsampled__]
        self.__struct_workspace__ = []
        self.__struct_idx_workspace__ = []
        return self

    def __next__(self):
        if self.it < self.n_batch:
            self.shuffle()
            bond, bond_onsite = self.get_bond()
            data = (bond, bond_onsite, self.get_env(), self.__struct_workspace__,
                    self.kpoint, self.eigen_list[self.__struct_idx_workspace__].astype(float))
            self.it += 1
            return data
        else:
            raise StopIteration

    def __len__(self):
        return self.n_batch

    def atom_rearrangement(self, input):
        # input
        pass

if __name__ == '__main__':
    from ase.build import graphene_nanoribbon
    from dptb.structure.structure import BaseStruct
    atoms = graphene_nanoribbon(1.5, 1, type='armchair', saturated=True)
    basestruct = BaseStruct(atom=atoms, format='ase', cutoff=1.5, proj_atom_anglr_m={'C': ['s', 'p']}, proj_atom_neles={"C":4})

    p = Processor(structure_list=[basestruct, basestruct, basestruct, basestruct], kpoint=1, eigen_list=[1,2,3,4], batchsize=1, env_cutoff=1.5)

    count = 0
    for data in p:
        print(count)
        count += 1
        if count == 2:
            break

    count = 0
    for data in p:
        print(count)
        count+=1