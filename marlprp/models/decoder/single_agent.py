import abc
from typing import Tuple, TypedDict

import torch
from torch.distributions import Categorical
from tensordict import TensorDict
from einops import rearrange


from marlprp.env.env import MSPRPEnv
from marlprp.env.instance import MSPRPState
from marlprp.utils.ops import gather_by_index
from marlprp.models.policy_args import MatNetParams
from marlprp.models.encoder.base import MatNetEncoderOutput

from .base import BaseDecoder
from .mlp import JobMachinePointer
from .attn import AttentionPointer


__all__ = []

class SingleAgentAction(TypedDict):
    idx: torch.Tensor  # the flat (in case of combined action space) action
    job: torch.Tensor
    machine: torch.Tensor



class HierarchicaDecoder(BaseDecoder):

    def __init__(self, model_params: MatNetParams):
        super().__init__(model_params)
        self.use_attn_mask = model_params.decoder_attn_mask
        self.dec_strategy = None
        self.shelf_pointer = AttentionPointer(model_params, decoder_type="shelf")
        self.sku_pointer = AttentionPointer(model_params, decoder_type="sku")

    def forward(self, embeddings: MatNetEncoderOutput, state: MSPRPState, env: MSPRPEnv, return_logp = False):
        node_mask = env.get_node_mask(state)
        if self.use_attn_mask:
            attn_mask = ~node_mask
        shelf_logits = self.shelf_pointer(embeddings, state, attn_mask=attn_mask)
        next_shelves, shelf_logps = self.dec_strategy.step(shelf_logits, node_mask, state, key="shelf")
        # partial step through the environment to get intermediate state s'
        intermediate_state = env.step(next_shelves, state)
        # get new action mask from intermediate state
        sku_mask = env.get_sku_mask(intermediate_state, next_shelves["shelf"])
        if self.use_attn_mask:
            attn_mask = ~sku_mask
        sku_logits = self.sku_pointer(embeddings, state, attn_mask=attn_mask)
        skus_to_pick, sku_logps = self.dec_strategy.step(sku_logits, sku_mask, intermediate_state, key="sku")

        # step through environment with handled conflicts to get next state
        next_state = env.step(skus_to_pick, intermediate_state)
        # save actions as tensordict
        actions = TensorDict({"shelf": next_shelves, "sku": skus_to_pick}, batch_size=next_shelves.batch_size)
        masks = TensorDict({"shelf": node_mask, "sku": sku_mask}, batch_size=state.batch_size)

        return_dict = {"state": state, "action": actions, "next": next_state, "action_mask": masks}
        if return_logp:
            logps = TensorDict({"shelf": shelf_logps, "sku": sku_logps}, batch_size=shelf_logps.batch_size)
            return_dict["logprobs"] = logps

        return TensorDict(return_dict, batch_size=state.batch_size, device=state.device)
    
    def get_logp_of_action(self, embeddings, actions: TensorDict, masks: TensorDict, state: MSPRPState):
        state = state.clone()
        masks = masks.clone()
        actions = actions.clone()

        shelf_action = actions["shelf"]
        shelf_mask = masks["shelf"]
        bs, n_agents, _ = shelf_mask.shape
        # unmask the "stay" action, where the agent simply stays at current node. This is 
        # selected in case of conflicts
        shelf_mask.scatter_(-1, state.current_location.view(bs, n_agents, 1), False)
        shelf_logp, shelf_entropy = self.shelf_decoder.get_logp_of_action(
            embeddings, action=shelf_action, mask=shelf_mask, state=state
        )
        # update state
        state.current_location = shelf_action["shelf"]

        sku_action = actions["sku"]
        sku_mask = masks["sku"]
        # unmask the "stay" action, where the agent simply stays at current node. This is 
        # selected in case of conflicts
        sku_mask[..., 0] = False
        sku_logp, sku_entropy = self.sku_decoder.get_logp_of_action(
            embeddings, action=sku_action, mask=sku_mask, state=state
        )

        action_logp = shelf_logp + sku_logp
        entropy = shelf_entropy + sku_entropy
        loss_mask = state.agent_pad_mask
        return action_logp, entropy, loss_mask
    
