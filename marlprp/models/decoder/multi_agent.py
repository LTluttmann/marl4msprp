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
from marlprp.utils.ops import gather_by_index, batchify
from marlprp.models.encoder.base import MatNetEncoderOutput

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

        self.shelf_decoder = MultiAgentShelfDecoder(model_params)
        self.sku_decoder = MultiAgentSkuDecoder(model_params)

    def forward(self, embeddings: MatNetEncoderOutput, state: MSPRPState, env: MSPRPEnv, return_logp = False):
        node_mask = env.get_node_mask(state)
        next_shelves, shelf_logps = self.shelf_decoder(embeddings, state, node_mask, env)

        # partial step through the environment to get intermediate state s'
        intermediate_state = env.step(next_shelves, state)
        # get new action mask from intermediate state
        sku_mask = env.get_sku_mask(intermediate_state, next_shelves)
        skus_to_pick, sku_logps = self.sku_decoder(embeddings, intermediate_state, sku_mask, env)

        # step through environment to get next state
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

        sku_logp, sku_entropy = self.sku_decoder.get_logp_of_action(
            embeddings, action=sku_action, mask=sku_mask, state=state
        )

        action_logp = shelf_logp + sku_logp
        entropy = shelf_entropy + sku_entropy
        loss_mask = state.agent_pad_mask
        return action_logp, entropy, loss_mask
    
    def _set_decode_strategy(self, decode_type, **kwargs):
        self.shelf_decoder._set_decode_strategy(decode_type, **kwargs)
        self.sku_decoder._set_decode_strategy(decode_type, **kwargs)

    def pre_decoding_hook(self, state, embeddings):
        self.shelf_decoder.dec_strategy.setup()
        self.sku_decoder.dec_strategy.setup()
        if self.shelf_decoder.dec_strategy.num_starts > 1:
            state = batchify(state, self.shelf_decoder.dec_strategy.num_starts)
            embeddings = batchify(embeddings, self.shelf_decoder.dec_strategy.num_starts)
        return state, embeddings
    
    def post_decoding_hook(self, state: MSPRPState, env: MSPRPEnv):
        shelf_logps, shelves, _ = self.shelf_decoder.post_decoding_hook(state, env)
        sku_logps, skus, state = self.sku_decoder.post_decoding_hook(state, env)
        logps = shelf_logps + sku_logps
        actions = TensorDict({"shelf": shelves, "sku": skus}, batch_size=state.batch_size)
        return logps, actions, state




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
        state: MSPRPState,
        mask: torch.Tensor,
        env: MSPRPEnv,
    ):
        num_agents = state.num_agents
        # get logits and mask
        logits = self.pointer(embeddings, state)
        mask = mask.clone()
        # initialize action buffer
        actions = []
        logps = []
        busy_agents = []
        while not mask.all():
            
            logp, step_mask = self._logits_to_logp(logits, mask)

            action, logp = self.dec_strategy.step(logp, step_mask, state)
            
            action = self._translate_action(action, state, env)
            actions.append(action)
            logps.append(logp)
            busy_agents.append(action["agent"])

            mask = self._update_mask(mask, action, state, busy_agents)

        # maybe pad to ensure all action buffers to have the same size
        n_active_agents = len(actions)
        # ((left/right padding of first dim; (left/right padding of second dim) 
        if self.pad:
            pad_size = [0, 0, 0, num_agents - n_active_agents]
        else:
            pad_size = [0, 0, 0, 0]

        actions = torch.stack(actions, dim=1)
        # NOTE we sort actions in ascending order of agent idx
        agent_sort_idx = torch.argsort(actions["agent"])
        actions = actions.gather(1, agent_sort_idx)
        actions = pad(actions, pad_size, value=0)
        # NOTE we sort logps in ascending order of agent idx
        logps = torch.stack(logps, dim=1)
        logps = logps.gather(1, agent_sort_idx)
        logps = F.pad(logps, pad_size, "constant", 0)

        return actions, logps

    @abc.abstractmethod
    def _logits_to_logp(self, logits, mask) -> Tuple[Tensor, Tensor]:
        pass


    @abc.abstractmethod
    def _update_mask(self, mask: Tensor, action: MultiAgentAction, state: MSPRPState, busy_agents: list) -> Tensor:
        pass


    @abc.abstractmethod
    def _translate_action(self, action_idx: torch.Tensor, state: MSPRPState, env: MSPRPEnv) -> MultiAgentAction:
        pass


    def get_logp_of_action(self,  embeddings: TensorDict, action: TensorDict, mask: torch.Tensor, state: MSPRPState):
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
            #(bs)
            dist_entropys = Categorical(probs=logp.exp()).entropy()            
        
        return selected_logp, dist_entropys


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

    def _translate_action(self, action_idx: torch.Tensor, state: MSPRPState, env: MSPRPEnv) -> MultiAgentAction:
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
    
    def _update_mask(self, mask: Tensor, action: MultiAgentAction, state, busy_agents: list) -> Tensor:
        bs, num_agents, num_nodes = mask.shape
        updated_mask = mask.clone()
        selected_shelf = action["shelf"]
        busy_agents = torch.stack(busy_agents, dim=1)

        updated_mask[..., state.num_depots:] = updated_mask.scatter(
            -1, 
            selected_shelf.view(bs, 1, 1).expand(-1, num_agents, 1), 
            True
        )[..., state.num_depots:]
        # okay now it gets wild. We have to make sure all idle agents can select some action. Therefore, we first need to determine 
        # which idle agents have no feasible action left (since all their feasible nodes were selected already)
        no_action_left = updated_mask.all(-1)
        updated_mask[no_action_left] = ~state.current_loc_ohe[no_action_left].bool()
        # updated_mask[..., :state.num_depots] = mask[..., :state.num_depots]
        updated_mask = updated_mask.scatter(-2, busy_agents.view(bs, -1, 1).expand(bs, -1, num_nodes), True)

        return updated_mask
    

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

    def _translate_action(self, action_idx: torch.Tensor, state: MSPRPState, env: MSPRPEnv) -> MultiAgentAction:
        # translate and store action
        selected_agent = action_idx % state.num_agents
        selected_sku = action_idx // state.num_agents
        action = TensorDict(
            {
                "idx": action_idx, 
                "agent": selected_agent, 
                "sku": selected_sku,
            },
            batch_size=state.batch_size
        )
        return action
    
    def _update_mask(self, mask: Tensor, action: MultiAgentAction, state: MSPRPState, busy_agents: list) -> Tensor:
        bs, n_agents, n_jobs = mask.shape

        busy_agents = torch.stack(busy_agents, dim=1)
        # NOTE below code doesnt help, since state.demand is not updated between agent updates
        # chosen_sku = action["sku"]
        # curr_agent = action["agent"]

        # pick_instance = chosen_sku != 0
        # full_demand_taken = torch.full_like(pick_instance, fill_value=False)
        # # subtract 1 for dummy item
        # sku = chosen_sku[pick_instance] - 1
        # agent = curr_agent[pick_instance]
        # remaining_capacity = state.remaining_capacity[pick_instance, agent]

        # # [BS, 1]
        # chosen_shelf = state.current_location.gather(1, curr_agent[:, None]).squeeze(1)
        # shelf_idx = chosen_shelf[pick_instance] - state.num_depots

        # # [num_visited]
        # demand_of_sku = state.demand[pick_instance, sku]
        # supply_at_shelf = state.supply[pick_instance, shelf_idx, sku]

        # taken_units = torch.min(demand_of_sku, supply_at_shelf)
        # taken_units = torch.min(taken_units, remaining_capacity)

        # full_demand_taken[pick_instance] = taken_units == demand_of_sku
        # mask[full_demand_taken].scatter(
        #     2, 
        #     chosen_sku[full_demand_taken].view(-1,1,1).expand(-1,n_agents,n_jobs), 
        #     True
        # )
        # mask[..., 0] = torch.where(mask.all(-1), False, mask[...,0])
        # mask = mask.scatter(-1, selected_job.view(bs, 1, 1).expand(-1, num_mas, 1), True)
        # mask[..., 0] = False
        mask = mask.scatter(-2, busy_agents.view(bs, -1, 1).expand(bs, -1, n_jobs), True)

        return mask
