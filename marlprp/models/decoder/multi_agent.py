import abc
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical
from einops import reduce, rearrange
from tensordict import TensorDict, pad
from typing import TypedDict, Tuple, List

from marlprp.env.env import MultiAgentEnv
from marlprp.env.instance import MSPRPState
from marlprp.models.policy_args import MahamParams
from marlprp.utils.ops import gather_by_index, batchify
from marlprp.models.encoder.base import MatNetEncoderOutput
from marlprp.decoding.strategies import DecodingStrategy, get_decoding_strategy

from .base import BaseDecoder
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

        self.dec_strategy = None
        self.shelf_decoder = MultiAgentShelfDecoder(model_params)
        self.sku_decoder = MultiAgentSkuDecoder(model_params)

    def forward(self, embeddings: MatNetEncoderOutput, state: MSPRPState, env: MultiAgentEnv, return_logp = False):
        node_mask = env.get_node_mask(state)
        next_shelves, shelf_logps = self.shelf_decoder(embeddings, state, node_mask, env)

        # partial step through the environment to get intermediate state s'
        intermediate_state = env.step(next_shelves, state)
        # get new action mask from intermediate state
        sku_mask = env.get_sku_mask(intermediate_state, next_shelves)
        skus_to_pick, sku_logps = self.sku_decoder(embeddings, intermediate_state, sku_mask, env)

        # handle conflicts
        conflicts = torch.logical_and(skus_to_pick["sku"] == 0, ~intermediate_state.agent_at_depot())
        current_nodes = state.current_location[conflicts]
        current_agent = conflicts.nonzero()[:, 1]
        current_idx = current_nodes * state.num_agents + current_agent
        current_action = TensorDict(
            {
                "idx": current_idx, 
                "agent": current_agent, 
                "shelf": current_nodes,
            },
            batch_size=current_nodes.shape
        )
        next_shelves[conflicts] = current_action
        shelf_logps[conflicts] = 0

        # step through environment with handled conflicts to get next state
        intermediate_state = env.step(next_shelves, state)
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
        assert (actions["shelf"]["agent"] == actions["sku"]["agent"]).all()

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
        state.current_location = state.current_location.scatter(
            1, shelf_action["agent"], shelf_action["shelf"]
        )

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
        loss_mask = state.agent_pad_mask.gather(1, actions["shelf"]["agent"])
        return action_logp, entropy, loss_mask
    
    def _set_decode_strategy(self, decode_type, **kwargs):
        self.dec_strategy = get_decoding_strategy(decode_type, **kwargs)
        self.shelf_decoder.dec_strategy = self.dec_strategy
        self.sku_decoder.dec_strategy = self.dec_strategy



class BaseMultiAgentDecoder(BaseDecoder):

    key = ...

    def __init__(self, pointer, params: MahamParams, dec_strategy) -> None:
        super().__init__(params)
        self.pointer = pointer
        self.eval_multistep = params.eval_multistep
        self.eval_per_agent = params.eval_per_agent
        self.pad = params.eval_multistep
        self.dec_strategy = dec_strategy
        self.loss = "listnet"

    def forward(
        self, 
        embeddings: MatNetEncoderOutput, 
        state: MSPRPState,
        mask: torch.Tensor,
        env: MultiAgentEnv,
    ):
        num_agents = state.num_agents
        # get logits and mask
        attn_mask = self.get_attn_mask(state, env)
        logits = self.pointer(embeddings, state, attn_mask=attn_mask)
        mask = mask.clone()
        # initialize action buffer
        actions = []
        logps = []
        busy_agents = []
        while not mask.all():
            
            logp, step_mask = self._logits_to_logp(logits, mask)

            action, logp = self.dec_strategy.step(logp, step_mask, state, key=self.key)
            
            action = self._translate_action(action, state, env)
            actions.append(action)
            logps.append(logp)
            busy_agents.append(action["agent"])

            mask = self._update_mask(mask, actions, state, busy_agents, env)

        # maybe pad to ensure all action buffers to have the same size
        n_active_agents = len(actions)
        # ((left/right padding of first dim; (left/right padding of second dim) 
        if self.pad:
            pad_size = [0, 0, 0, num_agents - n_active_agents]
        else:
            pad_size = [0, 0, 0, 0]

        actions = torch.stack(actions, dim=1)
        # NOTE we sort actions in ascending order of agent idx
        precedence = actions["agent"]
        agent_sort_idx = torch.argsort(precedence)
        actions = actions.gather(1, agent_sort_idx)
        actions["precedence"] = precedence
        actions = pad(actions, pad_size, value=0)
        # NOTE we sort logps in ascending order of agent idx
        logps = torch.stack(logps, dim=1)
        logps = logps.gather(1, agent_sort_idx)
        logps = F.pad(logps, pad_size, "constant", 0)

        return actions, logps

    def get_attn_mask(self, state, env):
        return None


    @abc.abstractmethod
    def _update_mask(self, mask: Tensor, actions: List[MultiAgentAction], state: MSPRPState, busy_agents: list, env) -> Tensor:
        pass


    @abc.abstractmethod
    def _translate_action(self, action_idx: torch.Tensor, state: MSPRPState, env: MultiAgentEnv) -> MultiAgentAction:
        pass


    def get_logp_of_action(self,  embeddings: TensorDict, action: TensorDict, mask: torch.Tensor, state: MSPRPState):
        if self.loss == "ce":
            return self._get_logp_of_action_ce(embeddings, action, mask, state)
        else:
            return self._get_logp_of_action_listnet(embeddings, action, mask, state)

    def _get_logp_of_action_ce(self,  embeddings: TensorDict, action: TensorDict, mask: torch.Tensor, state: MSPRPState):
        # get flat action indices
        action_indices = action["idx"].clone()

        # get logits and mask once
        logits = self.pointer(embeddings, state)
        # (bs, num_actions)
        logp, _ = self._logits_to_logp(logits, mask)

        #(bs, num_agents)
        selected_logp = gather_by_index(logp, action_indices, dim=1, squeeze=False)
        assert selected_logp.isfinite().all()
        # get entropy
        if self.eval_multistep:
            #(bs, num_agents)
            dist_entropys = Categorical(
                probs=rearrange(logp, "b (s a) -> b a s", a=state.num_agents).exp()
            ).entropy()
        else:
            #(bs, 1)
            dist_entropys = Categorical(probs=logp.exp()).entropy().unsqueeze(-1)
        
        return selected_logp, dist_entropys
    
    def _get_logp_of_action_listnet(self,  embeddings: TensorDict, action: TensorDict, mask: torch.Tensor, state: MSPRPState):
        # get flat action indices
        action_indices = action["idx"].clone()
        action_indices = action_indices.gather(1, action["precedence"])
        action_sort_idx = torch.argsort(action["precedence"])
        action_indices = action_indices.split(1, dim=1)
        # get logits and mask once
        logits = self.pointer(embeddings, state)
        bs, num_agents, num_actions = logits.shape
        selected_logps = []
        for rank, action_idx in enumerate(action_indices):
            agent = action["precedence"][:, rank]
            # (bs, num_actions)
            logp, _ = self._logits_to_logp(logits, mask)
            #(bs, 1)
            selected_logp = gather_by_index(logp, action_idx, dim=1, squeeze=False)
            selected_logps.append(selected_logp)
            # mask all predecessor actions for coming successor (like in listnet)
            mask.scatter_(1, agent[:, None, None].expand(bs,1,num_actions), True)

        selected_logps = torch.cat(selected_logps, dim=1)
        # reorder according to agent indices
        selected_logps = selected_logps.gather(1, action_sort_idx)
        assert selected_logps.isfinite().all()

        dist_entropys = Categorical(logits=logits).entropy()
        
        return selected_logps, dist_entropys

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

################################################################
#########################  DECODER #############################
################################################################


class MultiAgentShelfDecoder(BaseMultiAgentDecoder):

    key = "shelf"

    def __init__(self, params: MahamParams, dec_strategy = None, pointer = None) -> None:
        self.embed_dim = params.embed_dim
        if pointer is None:
            pointer = AttentionPointer(params, decoder_type=self.key)
        super().__init__(pointer=pointer, params=params, dec_strategy=dec_strategy)
        self.use_attn_mask = params.decoder_attn_mask

    def _translate_action(self, action_idx: torch.Tensor, state: MSPRPState, env: MultiAgentEnv) -> MultiAgentAction:
        # translate and store action
        selected_agent = action_idx % state.num_agents
        selected_shelf = action_idx // state.num_agents

        action = TensorDict(
            {
                "idx": action_idx, 
                "agent": selected_agent, 
                "shelf": selected_shelf,
            },
            batch_size=state.batch_size
        )
        return action
    
    def _update_mask(self, mask: Tensor, actions: List[MultiAgentAction], state, busy_agents: list, env: MultiAgentEnv) -> Tensor:
        action = actions[-1]

        bs, num_agents, num_nodes = mask.shape
        updated_mask = mask.clone()
        selected_shelf = action["shelf"]
        busy_agents = torch.stack(busy_agents, dim=1)

        # mask a shelf (not depot) if it got selected
        updated_mask[..., state.num_depots:] = updated_mask.scatter(
            -1, 
            selected_shelf.view(bs, 1, 1).expand(-1, num_agents, 1), 
            True
        )[..., state.num_depots:]

        if not env.is_multitour_instance:
            # in case of "one tour per agent" instanes (i.e. num_agents=None), an agent may only return to the depot if
            # all other active agents have enough capacity to pick all remaining items. If one agent returns to the depot,
            # we need to recalculate whether the other agents can handle the remaining demand 
            actions = torch.stack(actions, dim=-1)
            depot_selected = actions["shelf"].lt(state.num_depots)
            remaining_capacity = state.remaining_capacity.clone()
            going_to_depot = torch.full_like(state.agent_at_depot(), fill_value=False)
            going_to_depot.scatter_add_(1, actions["agent"], depot_selected)
            remaining_capacity.masked_fill_(going_to_depot, 0)
            remaining_capacity_expanded = remaining_capacity.unsqueeze(1).repeat(1, state.num_agents, 1)
            mask = torch.eye(state.num_agents, device=state.device).bool().unsqueeze(0).expand_as(remaining_capacity_expanded)
            # (bs, num_agents)
            capacity_of_other_agents = remaining_capacity_expanded.masked_fill(mask, 0).sum(-1)
            remaining_demand = state.demand.sum(-1, keepdim=True)
            mask_depot = capacity_of_other_agents.lt(remaining_demand) & remaining_demand.gt(0)
            # (bs, num_agents ,num_depots)
            mask_depot = mask_depot.unsqueeze(-1).repeat(1, 1, state.num_depots)
            updated_mask[..., :state.num_depots] = mask_depot
        # We have to make sure all idle agents can select some action. Therefore, we first need to determine 
        # which idle agents have no feasible action left (since all their feasible nodes were selected already)...
        no_action_left = updated_mask.all(-1)
        # and then we let these agents wait at their current location
        updated_mask[no_action_left] = ~state.current_loc_ohe[no_action_left].bool()
        # all actions of busy agents are masked
        updated_mask = updated_mask.scatter(-2, busy_agents.view(bs, -1, 1).expand(bs, -1, num_nodes), True)

        return updated_mask
    
    def get_attn_mask(self, state: MSPRPState, env: MultiAgentEnv):
        # in F.scaled_dot_product_attn, True mean "attend", False means not attend
        if self.use_attn_mask:  
            return ~env.get_node_mask(state)
    

class MultiAgentSkuDecoder(BaseMultiAgentDecoder):
    key = "sku"

    def __init__(self, params: MahamParams, dec_strategy = None, pointer = None) -> None:
        self.embed_dim = params.embed_dim
        if pointer is None:
            pointer = AttentionPointer(params, decoder_type=self.key)
        self.use_attn_mask = params.decoder_attn_mask
        super().__init__(pointer=pointer, params=params, dec_strategy=dec_strategy)

    def _translate_action(self, action_idx: torch.Tensor, state: MSPRPState, env: MultiAgentEnv) -> MultiAgentAction:
        batch_idx = torch.arange(action_idx.size(0), device=action_idx.device)
        # translate and store action
        selected_agent = action_idx % state.num_agents
        selected_sku = action_idx // state.num_agents

        loc_of_agent = gather_by_index(state.current_location, selected_agent)
        supply = state.supply_w_depot_and_dummy[batch_idx, loc_of_agent, selected_sku]
        capacity = gather_by_index(state.remaining_capacity, selected_agent)
        max_units = torch.minimum(supply, capacity)

        action = TensorDict(
            {
                "idx": action_idx, 
                "agent": selected_agent, 
                "sku": selected_sku,
                "max_units": max_units,
            },
            batch_size=state.batch_size
        )
        return action
    
    def _update_mask(self, mask: Tensor, actions: List[MultiAgentAction], state: MSPRPState, busy_agents: list, env) -> Tensor:
        bs, n_agents, n_jobs = mask.shape

        busy_agents = torch.stack(busy_agents, dim=1)

        action = actions[-1]
        chosen_sku = action["sku"]

        max_units_taken = torch.zeros_like(state.demand_w_dummy)
        for action in actions:
            max_units_taken.scatter_add_(1, index=action["sku"][:, None], src=action["max_units"][:, None])

        a = gather_by_index(max_units_taken, chosen_sku, dim=1)
        b = gather_by_index(state.demand_w_dummy, chosen_sku, dim=1)

        mask = torch.where(
            a.ge(b).view(bs, 1, 1).expand_as(mask), 
            mask.scatter(-1, chosen_sku.view(bs, 1, 1).expand(-1, n_agents, 1), True), 
            mask
        )
        mask[..., 0] = torch.where(mask[...,1:].all(-1), False, mask[...,0])
        mask = mask.scatter(-2, busy_agents.view(bs, -1, 1).expand(bs, -1, n_jobs), True)

        return mask

    def get_attn_mask(self, state: MSPRPState, env: MultiAgentEnv):
        # omit dummy item mask
        # in F.scaled_dot_product_attn, True mean "attend", False means not attend
        if self.use_attn_mask:  
            return ~env.get_sku_mask(state)
    