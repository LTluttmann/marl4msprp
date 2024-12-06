import abc
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical
from einops import reduce, rearrange
from tensordict import TensorDict, pad
from typing import TypedDict, Tuple

from marlprp.envs.base import Environment
from marlprp.utils.config import PolicyParams
from marlprp.utils.ops import gather_by_index
from marlprp.models.encoder.base import MatNetEncoderOutput, OperationsEncoderOutput

from .base import BaseDecoder, BasePointer
from .attn import MultiAgentAttentionPointer
from .mlp import JobMachinePointer, JobPointer


__all__ = [
    "MultiAgentAttnDecoder",
    "MultiAgentMLPDecoder",
    "MultiJobDecoder",
]


class MultiAgentAction(TypedDict):
    idx: Tensor
    job: Tensor
    machine: Tensor
    pad_mask: Tensor


class BaseMultiAgentDecoder(BaseDecoder):
    def __init__(self, pointer, params: PolicyParams) -> None:
        super().__init__(params)
        self.pointer = pointer
        self.eval_multistep = params.eval_multistep
        self.eval_per_agent = params.eval_per_agent
        self.pad = params.eval_multistep

    def forward(
        self, 
        embeddings: OperationsEncoderOutput, 
        td: TensorDict, 
        env: Environment, 
        return_logp: bool = False
    ):
        num_agents = env.num_mas
        # get logits and mask
        logits, mask = self.pointer(embeddings, td)
        mask = mask if mask is not None else ~td["action_mask"]
        # initialize action buffer
        actions = []
        busy_agents = []
        while not mask[...,1:].all():
            
            logp, step_mask = self._logits_to_logp(logits, mask)

            td = self.dec_strategy.step(logp, step_mask, td)
            
            action = self._translate_action(td, env)
            actions.append(action)
            busy_agents.append(action["machine"])

            mask = self._update_mask(mask, action, td, busy_agents)

        # maybe pad to ensure all action buffers to have the same size
        n_active_agents = len(actions)
        # ((left/right padding of first dim; (left/right padding of second dim) 
        if self.pad:
            pad_size = [0, 0, 0, num_agents - n_active_agents]
        else:
            pad_size = [0, 0, 0, 0]
        actions = torch.stack(actions, dim=1)
        actions = pad(actions, pad_size, value=0)
        if return_logp:
            logps = torch.stack(self.dec_strategy.logp["action"][-n_active_agents:], dim=1)
            logps = F.pad(logps, pad_size, "constant", 0)
            td["logprobs"] = logps
        # perform env step with all agent actions
        td.set("action", actions)

        return td
    
    @abc.abstractmethod
    def _logits_to_logp(self, logits, mask) -> Tuple[Tensor, Tensor]:
        pass


    @abc.abstractmethod
    def _update_mask(self, mask: Tensor, action: MultiAgentAction, td: TensorDict, busy_agents: list) -> Tensor:
        pass


    @abc.abstractmethod
    def _translate_action(self, td: TensorDict, env: Environment) -> MultiAgentAction:
        pass


    def get_logp_of_action(self,  embeddings: TensorDict, td: TensorDict):
        bs = td.size(0)
        # get flat action indices
        action_indices = td["action"]["idx"]
        pad_mask = td["action"]["pad_mask"]

        # get logits and mask once
        logits, mask = self.pointer(embeddings, td)
        # NOTE unmask "noop" to avoid nans and infs: in principle, every agent always has the chance to wait
        mask[..., 0] = False
        # (bs, num_actions)
        logp, _ = self._logits_to_logp(logits, mask)

        #(bs, num_agents)
        selected_logp = gather_by_index(logp, action_indices, dim=1)
        # get entropy
        if self.eval_multistep:
            #(bs, num_agents)
            dist_entropys = Categorical(logits=logits).entropy()
        else:
            #(bs)
            dist_entropys = Categorical(probs=logp.exp()).entropy()            
        

        loss_mask = ~pad_mask
        return selected_logp, dist_entropys, loss_mask


################################################################
###################### FJSP DECODER ############################
################################################################

class BaseMultiAgentDecoder4Heterogeneous(BaseMultiAgentDecoder):

    def __init__(self, pointer: BasePointer, params: PolicyParams) -> None:
        super().__init__(pointer=pointer, params=params)
        self.embed_dim = params.embed_dim

    def _logits_to_logp(self, logits, mask):
        if torch.is_grad_enabled() and self.eval_per_agent:
            # when training we evaluate on a per agent basis
            # perform softmax per agent
            logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
            # flatten logp for selection
            logp = rearrange(logp, "b m j -> b (j m)")

        else:
            # when rolling out, we sample iteratively from flattened prob dist
            mask = rearrange(mask, "b m j -> b (j m)")
            logits = rearrange(logits, "b m j -> b (j m)")
            logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
            
        return logp, mask

    def _translate_action(self, td: TensorDict, env: Environment) -> MultiAgentAction:
        # translate and store action
        action = td["action"]
        selected_machine = action % env.num_mas
        selected_job = action // env.num_mas
        action = TensorDict(
            {
                "idx": action, 
                "job": selected_job, 
                "machine": selected_machine,
                "pad_mask": torch.full_like(action, fill_value=True, dtype=torch.bool)
            },
            batch_size=td.batch_size
        )
        return action
    
    def _update_mask(self, mask: Tensor, action: MultiAgentAction, _, busy_agents: list) -> Tensor:
        bs, num_mas, num_jobs_plus_one = mask.shape

        busy_agents = torch.stack(busy_agents, dim=1)
        selected_job = action["job"]

        mask = mask.scatter(-1, selected_job.view(bs, 1, 1).expand(-1, num_mas, 1), True)
        mask[..., 0] = False
        mask = mask.scatter(-2, busy_agents.view(bs, -1, 1).expand(bs, -1, num_jobs_plus_one), True)

        return mask
    

class MultiAgentAttnDecoder(BaseMultiAgentDecoder4Heterogeneous):
    def __init__(self, params: PolicyParams) -> None:
        pointer = MultiAgentAttentionPointer(params)
        super().__init__(pointer=pointer, params=params)

    def pre_decoding_hook(self, td, env, embeddings: MatNetEncoderOutput):
        td, env, embeddings = super().pre_decoding_hook(td, env, embeddings)
        if not self.stepwise_encoding:
            self.pointer.compute_cache(embeddings["operations"])
        return td, env, embeddings
    

class MultiAgentMLPDecoder(BaseMultiAgentDecoder4Heterogeneous):
    def __init__(self, params: PolicyParams) -> None:
        pointer = JobMachinePointer(params)
        super().__init__(pointer=pointer, params=params)



##############################################
############# JSSP DECODER ###################
##############################################


class BaseMultiAgentDecoder4Homogeneous(BaseMultiAgentDecoder):
    def __init__(self, pointer: BasePointer, params: PolicyParams) -> None:
        super().__init__(pointer=pointer, params=params)
        
    def _logits_to_logp(self, logits, mask) -> Tuple[Tensor]:
        logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
        return logp
    
    def _update_mask(self, mask: Tensor, action: MultiAgentAction, td, _):

        job_next_ma_w_dummy = torch.cat(
            (torch.full_like(td["job_next_ma"][:, :1], -1), td["job_next_ma"]), 
            dim=1
        )

        job_where_ma_busy = job_next_ma_w_dummy == action["machines"][:, None]
        stop_action = action["job"] == 0

        # mask the selected job...
        mask = mask.scatter(-1, action["job"][:, None], True)
        # ...as well as jobs need to be executed on same machine
        mask[job_where_ma_busy] = True
        # mask all actions when wait action has been selected
        mask[stop_action, 1:] = True
        # always allow wait action to avoid nans
        mask[:, 0] = False
        return mask

    
    def _translate_action(self, td: TensorDict, env: Environment) -> MultiAgentAction:

        selected_job = td["action"]

        job_next_ma_w_dummy = torch.cat(
            (torch.full_like(td["job_next_ma"][:, :1], -1), td["job_next_ma"]), 
            dim=1
        )

        ma_of_job = job_next_ma_w_dummy.gather(1, selected_job[:, None]).squeeze(1)

        action = TensorDict(
            {
                "idx": selected_job, 
                "job": selected_job, 
                "machine": ma_of_job,
                "pad_mask": torch.full_like(action, fill_value=True, dtype=torch.bool)
            },
            batch_size=td.batch_size
        )
        return action


class MultiJobDecoder(BaseMultiAgentDecoder4Heterogeneous):
    def __init__(self, params: PolicyParams) -> None:
        pointer = JobPointer(params)
        super().__init__(pointer=pointer, params=params)

