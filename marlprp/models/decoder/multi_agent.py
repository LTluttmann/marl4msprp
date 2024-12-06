import abc
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical
from einops import reduce, rearrange
from tensordict import TensorDict, pad
from typing import TypedDict, Tuple

from marlprp.env.env import MSPRPEnv
from marlprp.env.instance import MSPRPState
from marlprp.utils.config import PolicyParams
from marlprp.utils.ops import gather_by_index
from marlprp.models.encoder.base import MatNetEncoderOutput

from .base import BaseDecoder, BasePointer
from .attn import AttentionPointer


__all__ = [
    "HierarchicalMultiAgentDecoder",
]


class MultiAgentAction(TypedDict):
    idx: Tensor
    agent: Tensor
    shelf: Tensor
    sku: Tensor



class HierarchicalMultiAgentDecoder(BaseDecoder):

    def __init__(self, model_params):
        super().__init__(model_params)

        self.shelf_decoder = MultiAgentShelfDecoder(model_params)
        self.sku_decoder = MultiAgentSkuDecoder(model_params)

    def forward(self, embeddings: MatNetEncoderOutput, tc: MSPRPState, env: MSPRPEnv, return_logp = False):
        node_mask = env.get_node_mask(tc)
        next_shelves = self.shelf_decoder(embeddings, node_mask, tc, env)
        item_mask = env.get_item_mask_from_node(tc, next_shelves)
        skus_to_pick = self.sku_decoder(embeddings, item_mask, tc, env)
        action = ...

    def get_logp_of_action(self, embeddings, td):
        shelf_logp, shelf_entropy, shelf_mask = self.shelf_decoder.get_logp_of_action(embeddings, td)
        sku_logp, sku_entropy, sku_mask = self.sku_decoder.get_logp_of_action(embeddings, td)

        action_logp = shelf_logp + sku_logp
        return action_logp





class BaseMultiAgentDecoder(BaseDecoder):
    def __init__(self, pointer, params: PolicyParams) -> None:
        super().__init__(params)
        self.pointer = pointer
        self.eval_multistep = params.eval_multistep
        self.eval_per_agent = params.eval_per_agent
        self.pad = params.eval_multistep

    def forward(
        self, 
        embeddings: MatNetEncoderOutput, 
        mask: torch.Tensor,
        tc: MSPRPState,
        env: MSPRPEnv, 
        return_logp: bool = False
    ):
        num_agents = tc.num_agents
        # get logits and mask
        logits, mask = self.pointer(embeddings, tc)
        mask = mask if mask is not None else ~tc["action_mask"]
        # initialize action buffer
        actions = []
        busy_agents = []
        while not mask[...,1:].all():
            
            logp, step_mask = self._logits_to_logp(logits, mask)

            tc = self.dec_strategy.step(logp, step_mask, tc)
            
            action = self._translate_action(tc, env)
            actions.append(action)
            busy_agents.append(action["agent"])

            mask = self._update_mask(mask, action, tc, busy_agents)

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
            tc["logprobs"] = logps

        # perform env step with all agent actions
        tc.set("action", actions)

        return tc
    
    @abc.abstractmethod
    def _logits_to_logp(self, logits, mask) -> Tuple[Tensor, Tensor]:
        pass


    @abc.abstractmethod
    def _update_mask(self, mask: Tensor, action: MultiAgentAction, td: TensorDict, busy_agents: list) -> Tensor:
        pass


    @abc.abstractmethod
    def _translate_action(self, td: TensorDict, env: MSPRPEnv) -> MultiAgentAction:
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
#########################  DECODER #############################
################################################################


class MultiAgentShelfDecoder(BaseMultiAgentDecoder):

    def __init__(self, params: PolicyParams) -> None:
        self.embed_dim = params.embed_dim
        pointer = AttentionPointer(params, decoder_type="shelf")
        super().__init__(pointer=pointer, params=params)

    def _logits_to_logp(self, logits, mask):
        if torch.is_grad_enabled() and self.eval_per_agent:
            # when training we evaluate on a per agent basis
            # perform softmax per agent
            logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
            # flatten logp for selection
            logp = rearrange(logp, "b a s -> b (s a)")

        else:
            # when rolling out, we sample iteratively from flattened prob dist
            mask = rearrange(mask, "b a s -> b (s a)")
            logits = rearrange(logits, "b a s -> b (s a)")
            logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
            
        return logp, mask

    def _translate_action(self, action_idx: torch.Tensor, tc: MSPRPState) -> MultiAgentAction:
        # translate and store action
        selected_agent = action_idx % tc.num_agents
        selected_shelf = action_idx // tc.num_agents
        action = TensorDict(
            {
                "idx": action_idx, 
                "agent": selected_agent, 
                "shelf": selected_shelf,
                "pad_mask": torch.full_like(action, fill_value=True, dtype=torch.bool)
            },
            batch_size=tc.batch_size
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
    

class MultiAgentSkuDecoder(BaseMultiAgentDecoder):

    def __init__(self, params: PolicyParams) -> None:
        self.embed_dim = params.embed_dim
        pointer = AttentionPointer(params, decoder_type="sku")
        super().__init__(pointer=pointer, params=params)

    def _logits_to_logp(self, logits, mask):
        if torch.is_grad_enabled() and self.eval_per_agent:
            # when training we evaluate on a per agent basis
            # perform softmax per agent
            logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
            # flatten logp for selection
            logp = rearrange(logp, "b a s -> b (s a)")

        else:
            # when rolling out, we sample iteratively from flattened prob dist
            mask = rearrange(mask, "b a s -> b (s a)")
            logits = rearrange(logits, "b a s -> b (s a)")
            logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
            
        return logp, mask

    def _translate_action(self, action_idx: torch.Tensor, tc: MSPRPState) -> MultiAgentAction:
        # translate and store action
        selected_agent = action_idx % tc.num_agents
        selected_sku = action_idx // tc.num_agents
        action = TensorDict(
            {
                "idx": action_idx, 
                "agent": selected_agent, 
                "sku": selected_sku,
                "pad_mask": torch.full_like(action, fill_value=True, dtype=torch.bool)
            },
            batch_size=tc.batch_size
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
