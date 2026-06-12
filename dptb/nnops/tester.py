import torch
import logging
from dptb.utils.tools import get_lr_scheduler, \
get_optimizer, j_must_have
from dptb.nnops.base_tester import BaseTester
from typing import Union, Optional
from dptb.data import AtomicDataset, DataLoader, AtomicData
from dptb.nn import build_model
from dptb.nnops.loss import Loss
from dptb.nnops.flow import HamiltonianCFM

log = logging.getLogger(__name__)
#TODO: complete the log output for initilizing the trainer

class Tester(BaseTester):

    def __init__(
            self,
            test_options: dict,
            common_options: dict,
            model: torch.nn.Module,
            test_datasets: AtomicDataset,
            flow_options: Optional[dict] = None,
            ) -> None:
        super(Tester, self).__init__(dtype=common_options["dtype"], device=common_options["device"])
        
        # init the object
        self.model = model.to(self.device)
        self.common_options = common_options
        self.test_options = test_options
        
        self.test_datasets = test_datasets

        self.test_loader = DataLoader(dataset=self.test_datasets, batch_size=test_options["batch_size"], shuffle=False)

        # loss function
        self.test_lossfunc = Loss(**test_options["loss_options"]["test"], **common_options, idp=self.model.hamiltonian.idp)
        flow_options = dict(flow_options or {})
        if flow_options.get("enabled", False):
            flow_options["validation_ode_steps"] = test_options.get(
                "flow_ode_steps",
                flow_options.get("validation_ode_steps", [1, 3, 10]),
            )
        self.flow_cfm = HamiltonianCFM(
            flow_options,
            idp=self.model.hamiltonian.idp,
            dtype=self.dtype,
            device=self.device,
        )
        self.log_direct_target_fed_loss = bool(
            test_options.get("log_direct_target_fed_loss", True)
        )

    def build(self):
        return self

    @staticmethod
    def _loss_component_state(lossfunc, *, prefix="test"):
        loss_obj = getattr(lossfunc, "loss", lossfunc)
        state = {}
        onsite_comp = getattr(loss_obj, "last_onsite_loss", None)
        hopping_comp = getattr(loss_obj, "last_hopping_loss", None)
        if onsite_comp is not None:
            state[f"{prefix}_onsite_loss"] = onsite_comp
        if hopping_comp is not None:
            state[f"{prefix}_hopping_loss"] = hopping_comp
        return state
    
    def iteration(self, batch):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        self.model.eval()
        batch = batch.to(self.device)
        
        # record the batch_info to help reconstructing sub-graph from the batch
        batch_info = {
            "__slices__": batch.__slices__,
            "__cumsum__": batch.__cumsum__,
            "__cat_dims__": batch.__cat_dims__,
            "__num_nodes_list__": batch.__num_nodes_list__,
            "__data_class__": batch.__data_class__,
        }

        batch = AtomicData.to_AtomicDataDict(batch)

        batch_for_loss = batch.copy() # make a shallow copy in case the model change the batch data
        if self.flow_cfm.enabled:
            state = {'field': 'iteration'}
            if self.log_direct_target_fed_loss:
                direct_batch = batch.copy()
                like = direct_batch.get(
                    self.flow_cfm.node_target_key,
                    direct_batch.get(self.flow_cfm.edge_target_key),
                )
                direct_batch[self.flow_cfm.flow_time_key] = torch.ones(
                    self.flow_cfm._num_graphs(direct_batch),
                    device=like.device,
                    dtype=like.dtype,
                )
                direct_pred = self.model(direct_batch)
                direct_pred.update(batch_info)
                batch_for_loss.update(batch_info)
                direct_loss = self.test_lossfunc(direct_pred, batch_for_loss)
                state["test_direct_target_fed_loss"] = direct_loss.detach()
                state.update(
                    self._loss_component_state(
                        self.test_lossfunc,
                        prefix="test_direct_target_fed",
                    )
                )

            primary_loss = None
            for num_steps in self.flow_cfm.validation_ode_steps:
                sampled = self.flow_cfm.sample(self.model, batch, num_steps=num_steps)
                sampled.update(batch_info)
                batch_for_loss.update(batch_info)
                sample_loss = self.test_lossfunc(sampled, batch_for_loss)
                prefix = f"test_cfm_euler_{num_steps}"
                state[f"{prefix}_loss"] = sample_loss.detach()
                component_state = self._loss_component_state(self.test_lossfunc, prefix=prefix)
                state.update(component_state)
                if primary_loss is None:
                    primary_loss = sample_loss
                    for component in ("onsite_loss", "hopping_loss"):
                        source_key = f"{prefix}_{component}"
                        if source_key in component_state:
                            state[f"test_{component}"] = component_state[source_key]
            if primary_loss is None:
                raise ValueError("CFM testing requires at least one positive flow_ode_steps value.")
            state["test_loss"] = primary_loss.detach()
            self.call_plugins(queue_name='iteration', time=self.iter, **state)
            self.iter += 1
            return primary_loss.detach()

        #TODO: the rescale/normalization can be added here
        batch = self.model(batch)

        #TODO: this could make the loss function unjitable since t he batchinfo in batch and batch_for_loss does not necessarily 
        #       match the torch.Tensor requiresment, should be improved further

        batch.update(batch_info)
        batch_for_loss.update(batch_info)

        loss = self.test_lossfunc(batch, batch_for_loss)

        state = {'field':'iteration', "test_loss": loss.detach()}
        state.update(self._loss_component_state(self.test_lossfunc, prefix="test"))
        self.call_plugins(queue_name='iteration', time=self.iter, **state)
        self.iter += 1

        return loss.detach()
    
    def epoch(self) -> None:

        for ibatch in self.test_loader:
            # iter with different structure
            self.iteration(ibatch)
