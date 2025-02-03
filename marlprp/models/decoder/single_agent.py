from typing import TypedDict

import torch
from einops import rearrange
from tensordict import TensorDict
from torch.distributions import Categorical

from marlprp.env.env import MultiAgentEnv
from marlprp.env.instance import MSPRPState
from marlprp.models.policy_args import TransformerParams
from marlprp.models.encoder.base import MatNetEncoderOutput

from .base import BaseDecoder
from .attn import AttentionPointer


__all__ = []

class SingleAgentAction(TypedDict):
    agent: torch.Tensor  # the flat (in case of combined action space) action
    shelf: torch.Tensor
    sku: torch.Tensor



class HierarchicalSingleAgentDecoder(BaseDecoder):

    def __init__(self, model_params: TransformerParams):
        super().__init__(model_params)
        self.use_attn_mask = model_params.decoder_attn_mask
        self.dec_strategy = None
        self.shelf_pointer = AttentionPointer(model_params, decoder_type="shelf")
        self.sku_pointer = AttentionPointer(model_params, decoder_type="sku")

    def forward(self, embeddings: MatNetEncoderOutput, state: MSPRPState, env: MultiAgentEnv, return_logp = False):
        shelf_mask = env.get_node_mask(state)
        attn_mask = self._get_attn_mask(shelf_mask, state)
        shelf_logits = self.shelf_pointer(embeddings, state, attn_mask=attn_mask)
        shelf_logp, step_mask = self._logits_to_logp(shelf_logits, shelf_mask)
        next_shelves, shelf_logps = self.dec_strategy.step(shelf_logp, step_mask, state, key="shelf")
        shelf_action = self._translate_action(next_shelves, state, "shelf")

        # partial step through the environment to get intermediate state s'
        intermediate_state = env.step(shelf_action, state)
        # get new action mask from intermediate state
        sku_mask = env.get_sku_mask(intermediate_state, shelf_action)
        attn_mask = self._get_attn_mask(sku_mask, state)
        sku_logits = self.sku_pointer(embeddings, intermediate_state, attn_mask=attn_mask)
        sku_logp, step_mask = self._logits_to_logp(sku_logits, sku_mask)
        skus_to_pick, sku_logps = self.dec_strategy.step(sku_logp, step_mask, intermediate_state, key="sku")
        skus_action = self._translate_action(skus_to_pick, state, "sku") 

        # step through environment with handled conflicts to get next state
        next_state = env.step(skus_action, intermediate_state)

        # save actions as tensordict
        actions = TensorDict({
            "shelf": shelf_action, 
            "sku": skus_action
        }, batch_size=shelf_action.batch_size)
        masks = TensorDict({"shelf": shelf_mask, "sku": sku_mask}, batch_size=state.batch_size)

        return_dict = {"state": state, "action": actions, "next": next_state, "action_mask": masks}
        if return_logp:
            logps = TensorDict({"shelf": shelf_logps, "sku": sku_logps}, batch_size=shelf_logps.batch_size)
            return_dict["logprobs"] = logps

        return TensorDict(return_dict, batch_size=state.batch_size, device=state.device)
    
    def _translate_action(self, action_idx: torch.Tensor, state: MSPRPState, key: str):
        action = TensorDict(
            {
                "agent": state.active_agent,
                key: action_idx.unsqueeze(1),
                "idx": action_idx.unsqueeze(1),
            }, 
            batch_size=(*state.batch_size, 1)
        )
        return action
    
    def _get_attn_mask(self, action_mask: torch.Tensor, state: MSPRPState):
        bs, n_agents, n_actions = action_mask.shape
        if self.use_attn_mask:
            attn_mask = ~action_mask.gather(1, state.active_agent.view(bs, 1, 1).expand(bs, n_actions))
        else:
            attn_mask = None
        return attn_mask

    def get_logp_of_action(self, embeddings, actions: TensorDict, masks: TensorDict, state: MSPRPState):
        state = state.clone()
        masks = masks.clone()
        actions = actions.clone()
    
        shelf_action = actions["shelf"]
        shelf_mask = masks["shelf"]

        if self.use_attn_mask:
            attn_mask = ~shelf_mask
        else:
            attn_mask = ~shelf_mask

        shelf_logits = self.shelf_pointer(embeddings, state, attn_mask=attn_mask)
        shelf_logp, _ = self._logits_to_logp(shelf_logits, shelf_mask)
        shelf_entropy = Categorical(probs=shelf_logp.exp()).entropy().unsqueeze(-1)
        selected_shelf_logp = shelf_logp.gather(1, shelf_action["idx"])

        # update state
        state.current_location = state.current_location.scatter(
            1, shelf_action["agent"], shelf_action["shelf"]
        )

        sku_action = actions["sku"]
        sku_mask = masks["sku"]

        if self.use_attn_mask:
            attn_mask = ~sku_mask
        else:
            attn_mask = ~sku_mask

        sku_logits = self.sku_pointer(embeddings, state, attn_mask=attn_mask)
        sku_logp, _ = self._logits_to_logp(sku_logits, sku_mask)
        sku_entropy = Categorical(probs=sku_logp.exp()).entropy().unsqueeze(-1)
        selected_sku_logp = sku_logp.gather(1, sku_action["idx"])

        action_logp = selected_shelf_logp + selected_sku_logp
        entropy = shelf_entropy + sku_entropy

        return action_logp, entropy, None

    def _logits_to_logp(self, logits, mask):
        bs = logits.size(0)
        logits = logits.view(bs, -1)
        mask = mask.view(bs, -1)
        # (bs, num_actions)
        logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)  
        return logp, mask
    

class Hierarchical2dPtrDecoder(HierarchicalSingleAgentDecoder):
    def __init__(self, model_params: TransformerParams):
        super().__init__(model_params)

    def _translate_action(self, action_idx: torch.Tensor, state: MSPRPState, key: str):
        # translate and store action
        selected_agent = action_idx % state.num_agents
        selected_action = action_idx // state.num_agents

        action = TensorDict(
            {
                "idx": action_idx.unsqueeze(1), 
                "agent": selected_agent.unsqueeze(1), 
                key: selected_action.unsqueeze(1),
            },
            batch_size=(*state.batch_size, 1)
        )
        return action
    
    def _logits_to_logp(self, logits, mask):
        mask = rearrange(mask, "b a s -> b (s a)")
        logits = rearrange(logits, "b a s -> b (s a)")
        logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
        return logp, mask