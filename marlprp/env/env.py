from typing import Optional, Union

import torch
import numpy as np
import torch.nn.functional as F
from tensordict import TensorDict

from marlprp.utils.utils import Registry
from marlprp.env.instance import MSPRPState
from marlprp.utils.ops import gather_by_index
from marlprp.env.generator import MSPRPGenerator
from marlprp.utils.config import EnvParams, EnvParamList


env_registry = Registry()







@env_registry.register("msprp")
class MultiAgentEnv:

    name = "msprp"

    def __init__(self, params: Union[EnvParams, EnvParamList] = None) -> None:
        if isinstance(params, EnvParamList):
            self.generators = [MSPRPGenerator(param) for param in params]
            self.params = params[0]
        else:
            self.generators = [MSPRPGenerator(params)]
            self.params = params
        # when num agents is None, we assume there are as many agents as needed to collect
        # all demanded items in one go (i.e. one tour per picker / agent)
        self.is_multitour_instance = self.params.num_agents is not None
        self.generators = sorted(self.generators, key=lambda x: x.size)

    @classmethod
    def initialize(cls, params: EnvParams):
        EnvCls = env_registry.get(params.name)
        return EnvCls(params)

    def reset(self, td: Optional[TensorDict] = None, batch_size=None) -> MSPRPState:
        """Reset function to call at the beginning of each episode"""
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
        if td is None or td.is_empty():
            generator = np.random.choice(self.generators)
            td = generator(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._reset(td)

    def _reset(self, td: TensorDict) -> MSPRPState:

        state = MSPRPState.initialize(
            **td.clone(),
        )
        return state

    def step(self, actions: torch.Tensor, state: MSPRPState) -> MSPRPState:
        state = state.clone()
        precedence = actions.get("precedence", None)
        if precedence is not None:
            actions = actions.gather(1, precedence)
        
        actions = actions.split(1, dim=1)
        for action in actions:
            action = action.squeeze(1)
            agent = action["agent"]
            shelf = action.get("shelf", None)
            sku = action.get("sku", None)
            units = action.get("units", None)

            if shelf is not None:
                state = self._update_from_shelf(state, shelf, current_agent=agent)
                #state.sku_mask = self.get_sku_mask(state)

            if sku is not None:
                state = self._update_from_sku(state, sku, curr_agent=agent, units=units)
                #state.shelf_mask = self.get_node_mask(state)

        return state

    def _update_from_shelf(self, state: MSPRPState, next_node: torch.Tensor, current_agent: torch.Tensor):
        """
        :param chosen_shelf: [BS]
        """
        bs = next_node.size(0)
        batch_idx = torch.arange(0, bs, device=state.device)
        # [BS, 1, 2]
        curr_location = state.current_location.gather(1, current_agent[:, None])
        curr_coord = state.coordinates.gather(1, curr_location[:, None].expand(-1,1,2)).squeeze(1)
        new_coord = state.coordinates.gather(1, next_node[:, None, None].expand(-1,1,2)).squeeze(1)
        step_dist = (curr_coord - new_coord).norm(p=2, dim=-1)  # (batch_dim)

        state.tour_length[batch_idx, current_agent] += step_dist

        state.current_location[batch_idx, current_agent] = next_node

        # if agent returns to packing station, record how many items need to be packed at packing station
        returns_to_depot = torch.logical_and(next_node.lt(state.num_depots), ~curr_location.lt(state.num_depots).squeeze(1))
        depot_id = next_node[returns_to_depot]
        depot_agent = current_agent[returns_to_depot]
        load = state.capacity - gather_by_index(state.remaining_capacity, current_agent, dim=1)
        state.packing_items[returns_to_depot, depot_id] += load[returns_to_depot]
        if self.is_multitour_instance:
            # reset remaining load when at depot
            state.remaining_capacity[returns_to_depot, depot_agent] = state.capacity
        else:
            # set remaining load to zero to prevent the agent from leaving the depot again
            state.remaining_capacity[returns_to_depot, depot_agent] = 0

        return state

    def _update_from_sku(self, state: MSPRPState, chosen_sku: torch.Tensor, curr_agent: torch.Tensor, units: torch.Tensor = None):
        """
        :param chosen_sku: [BS]
        """
        # Update the dynamic elements differently for if we visit depot vs. a city
        # NOTE: Need to use the curr_node property instead of the shelves determined by the actor, since
        # during beam search different shelves might occur as beam parents for the sku child nodes
        pick_instance = chosen_sku != 0

        # subtract 1 for dummy item
        chosen_sku = chosen_sku[pick_instance] - 1
        agent = curr_agent[pick_instance]
        remaining_capacity = state.remaining_capacity[pick_instance, agent]

        # [BS, 1]
        chosen_shelf = state.current_location.gather(1, curr_agent[:, None]).squeeze(1)
        shelf_idx = chosen_shelf[pick_instance] - state.num_depots

        # [num_visited]
        demand_of_sku = state.demand[pick_instance, chosen_sku]
        supply_at_shelf = state.supply[pick_instance, shelf_idx, chosen_sku]

        if units is not None:
            taken_units = units[pick_instance]
            assert taken_units.le(torch.min(torch.min(demand_of_sku, supply_at_shelf), remaining_capacity)).all()
        else:
            taken_units = torch.min(demand_of_sku, supply_at_shelf)
            taken_units = torch.min(taken_units, remaining_capacity)

        # state.zero_units_taken[pick_instance, agent] = taken_units.eq(0).float()

        state.supply[pick_instance, shelf_idx, chosen_sku] -= taken_units
        state.remaining_capacity[pick_instance, agent] -= taken_units   
        state.demand[pick_instance, chosen_sku] -= taken_units

        return state

    def get_node_mask(self, state: MSPRPState):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        # (bs, num_agents)
        has_no_capacity = state.remaining_capacity.eq(0)
        # (bs, num_shelves)
        mask_loc = (state.demand[:, None, :].lt(1e-6) |  state.supply.lt(1e-6)).all(-1)
        # (bs, num_agents, num_shelves)
        mask_loc_per_agent = has_no_capacity[..., None] | mask_loc[:, None]
        no_more_demand = state.demand.eq(0).all(1)

        if not self.is_multitour_instance:
            # (bs, num_agents, num_agents)
            remaining_capacity_expanded = state.remaining_capacity.unsqueeze(1).repeat(1, state.num_agents, 1)
            mask = torch.eye(state.num_agents, device=state.device).bool().unsqueeze(0).expand_as(remaining_capacity_expanded)
            # (bs, num_agents)
            capacity_of_other_agents = remaining_capacity_expanded.masked_fill(mask, 0).sum(-1)
            remaining_demand = state.demand.sum(-1, keepdim=True)
            mask_depot = capacity_of_other_agents.lt(remaining_demand) & remaining_demand.gt(0)
            # (bs, num_agents ,num_depots)
            mask_depot = mask_depot.unsqueeze(-1).repeat(1, 1, state.num_depots)
            # NOTE this check should not be necessary
            # mask_depot = ~mask_loc_per_agent.all(-1, keepdims=True).repeat(1, 1, state.num_depots)
        else:
            # We should avoid traveling to the depot back-to-back, except instance is done
            # (bs, num_agents)
            mask_depot = state.agent_at_depot() & ~no_more_demand[:, None]
            # (bs, num_agents, num_depots)
            mask_depot = mask_depot[..., None].repeat(1, 1, state.num_depots)
            
        # for finished (i.e. no demand and back at depot) instances, mask all but current node
        finished = state.agent_at_depot() & no_more_demand[:, None]
        mask_depot[finished] = ~(state.current_loc_ohe[...,:state.num_depots][finished].bool())
        # (bs, num_agents, num_nodes)
        agent_node_mask = torch.cat((mask_depot, mask_loc_per_agent), 2).bool()
        # padded agents just stay where they are
        agent_node_mask[state.agent_pad_mask] = ~(state.current_loc_ohe[state.agent_pad_mask].bool())
        # if True:
        #     curr_shelf_has_supply = ~gather_by_index(agent_node_mask, state.current_location, 2)
        #     keep_agent_at_shelf = curr_shelf_has_supply & state.current_location != 0
        #     agent_node_mask[keep_agent_at_shelf] = ~(state.current_loc_ohe[keep_agent_at_shelf].bool())
        return agent_node_mask


    def get_sku_mask(self, state: MSPRPState, chosen_nodes: TensorDict = None):
        if chosen_nodes is None:
            return state.demand.eq(0)
        else:
            return self.get_sku_mask_at_node(state, chosen_nodes)

    def get_sku_mask_at_node(self, state: MSPRPState, chosen_nodes: TensorDict = None):
        """
        Gets a (batch_size, n_items+1) mask with the feasible actions depending on item supply at visited 
        shelf and item demand. 0 = feasible, 1 = infeasible
        NOTE that this returns mask of shape (bs, num_items+1) for one additional dummy item in case no 
        item is feasible (ie when at the depot). This would raise issues with softmax
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return: mask (batch_size, n_items) with 0 = feasible, 1 = infeasible
        """
        # [BS]
        # lets assume that, after selecting a shelf, we first update the state and then
        # select an item. This is much cleaner i think
        if chosen_nodes is None:
            chosen_nodes = TensorDict(
                {
                    "shelf": state.current_location,
                    "agent": torch.arange(state.num_agents, device=state.device, dtype=torch.long)
                }, 
                batch_size=state.batch_size
            )

        chosen_nodes = chosen_nodes.split(1, dim=1)

        # sku_masks = []
        batch_idx = torch.arange(state.size(0), device=state.device, dtype=torch.long)
        sku_masks = torch.full(size=(state.size(0), state.num_agents, state.num_skus+1), fill_value=True, device=state.device)
        for node_action in chosen_nodes:

            node = node_action["shelf"].squeeze(1)
            agent = node_action["agent"].squeeze(1)

            depot = node.lt(state.num_depots) # [BS]
            visit = ~depot # [BS]
            # [BS, 1]
            shelf = node[visit, None, None] - state.num_depots
            # [bs, skus]
            supply_at_chosen_node = state.supply[visit].gather(
                1, shelf.expand(-1, 1, state.num_skus)
            ).squeeze(1)
            # [bs, skus]
            supply_mask = supply_at_chosen_node.eq(0)
            # [bs, skus]
            demand_mask = state.demand[visit].eq(0)

            mask = torch.zeros_like(state.demand, dtype=torch.bool)
            mask[depot] = True
            mask[visit] = torch.logical_or(supply_mask, demand_mask)

            # add one dimension for a dummy item which can be chosen when at the depot
            mask = torch.cat((visit[:,None], mask), dim=1)
            # in case of conflicts, a picker may stay at the current position. In this case, probably all
            # skus are taken from the shelf already, making all actions infeasible. Here, we detect these
            # cases and unmask the dummy item
            mask[:, 0] = torch.where(mask.all(-1), False, mask[:,0])

            sku_masks[batch_idx, agent] =  mask
            # sku_masks.scatter(1, agent.view(-1, 1, 1).expand(-1, 1, state.num_skus+1), mask[:, None])
            # sku_masks.append(mask)

        # sku_masks = torch.stack(sku_masks, dim=1)
        # NOTE no mask for padded agents needed, as they stay at a depot and thus always select the dummy item by above logic
        return sku_masks
    

    def get_reward(self, state: MSPRPState, mode: str = "val"):
        if self.params.goal == "min-max":
            distance = state.tour_length.max(1).values
        else:
            distance = state.tour_length.sum(1)

        # calc entropy over packing station utilization
        logp = F.log_softmax(state.packing_items, dim=-1)
        entropy = -(logp.exp() * logp).sum(1)

        reward = distance + self.params.packing_ratio_penalty * entropy

        return -reward



@env_registry.register("ar")
class AREnv(MultiAgentEnv):
    name = "ar"

    def __init__(self, params = None):
        super().__init__(params)
        assert self.params.num_agents is None, "equity transformer expects one agent per tour"
        assert not self.is_multitour_instance, "equity transformer expects one agent per tour"


    def _reset(self, td: TensorDict) -> MSPRPState:
        td["active_agent"] = torch.zeros((*td.batch_size, 1), device=td.device, dtype=torch.long)
        state = MSPRPState.initialize(
            **td.clone(),
        )
        return state

    def _update_from_shelf(self, state: MSPRPState, next_node: torch.Tensor, current_agent: torch.Tensor):
        """
        :param chosen_shelf: [BS]
        """
        assert (state.active_agent == current_agent[:, None]).all()
        bs = next_node.size(0)
        batch_idx = torch.arange(0, bs, device=state.device)
        # [BS, 1, 2]
        curr_location = state.current_location.gather(1, current_agent[:, None])
        curr_coord = state.coordinates.gather(1, curr_location[:, None].expand(-1,1,2)).squeeze(1)
        new_coord = state.coordinates.gather(1, next_node[:, None, None].expand(-1,1,2)).squeeze(1)
        step_dist = (curr_coord - new_coord).norm(p=2, dim=-1)  # (batch_dim)

        state.tour_length[batch_idx, current_agent] += step_dist

        state.current_location[batch_idx, current_agent] = next_node

        # if agent returns to packing station, record how many items need to be packed at packing station
        returns_to_depot = torch.logical_and(next_node.lt(state.num_depots), ~curr_location.lt(state.num_depots).squeeze(1))
        depot_id = next_node[returns_to_depot]
        depot_agent = current_agent[returns_to_depot]
        load = state.capacity - gather_by_index(state.remaining_capacity, current_agent, dim=1)
        state.packing_items[returns_to_depot, depot_id] += load[returns_to_depot]

        state.remaining_capacity[returns_to_depot, depot_agent] = 0
        state.active_agent[returns_to_depot] += 1
        state.active_agent.clamp_max_(state.num_valid_agents[:, None]-1).clamp_min_(0)
        return state

    def get_node_mask(self, state: MSPRPState):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        # (bs, 1)
        current_agent = state.active_agent
        current_agent_position = state.current_location.gather(1, current_agent)

        remaining_agents = state.num_valid_agents - (current_agent.squeeze(1) + 1)
        remaining_agent_capacity = remaining_agents * state.capacity
        # (bs, 1)
        has_no_capacity = state.remaining_capacity.gather(1, current_agent).eq(0)
        # (bs, num_shelves)
        mask_loc = (state.demand[:, None, :].lt(1e-6) |  state.supply.lt(1e-6)).all(-1)
        # (bs, 1, num_shelves)
        mask_loc_and_agent = has_no_capacity[..., None] | mask_loc[:, None]
        
        remaining_demand = state.demand.sum(-1)
        mask_depot = remaining_agent_capacity.lt(remaining_demand) & remaining_demand.gt(0)
        mask_depot = mask_depot.view(-1, 1, 1).repeat(1, 1, state.num_depots)

        # for finished (i.e. no demand and back at depot) instances, mask all but current node
        no_more_demand = state.demand.eq(0).all(1)
        finished = current_agent_position.lt(state.num_depots) & no_more_demand[:, None]
        finished_agents_positions = current_agent_position[finished]
        mask_depot[finished] = mask_depot[finished].scatter(-1, finished_agents_positions[:, None], False)
        # (bs, num_agents, num_nodes)
        agent_node_mask = torch.cat((mask_depot, mask_loc_and_agent), 2).bool()
        return agent_node_mask
        


@env_registry.register("sa")
class SingleAgentSteppingEnv(MultiAgentEnv):
    name = "sa"

    def __init__(self, params = None):
        """Env for 2d-Ptr, where we observe M agents per timestep, but only generate an action for one of them"""
        super().__init__(params)
        assert not self.is_multitour_instance, "expects one agent per tour"

    def get_node_mask(self, state: MSPRPState):
        # (bs, num_agents)
        has_no_capacity = state.remaining_capacity.eq(0)
        # (bs, num_shelves)
        mask_loc = (state.demand[:, None, :].lt(1e-6) |  state.supply.lt(1e-6)).all(-1)
        # (bs, num_agents, num_shelves)
        mask_loc_per_agent = has_no_capacity[..., None] | mask_loc[:, None]
        no_more_demand = state.demand.eq(0).all(1)

        # (bs, num_agents, num_agents)
        remaining_capacity_expanded = state.remaining_capacity.unsqueeze(1).repeat(1, state.num_agents, 1)
        mask = torch.eye(state.num_agents, device=state.device).bool().unsqueeze(0).expand_as(remaining_capacity_expanded)
        # (bs, num_agents)
        capacity_of_other_agents = remaining_capacity_expanded.masked_fill(mask, 0).sum(-1)
        remaining_demand = state.demand.sum(-1, keepdim=True)
        mask_depot = capacity_of_other_agents.lt(remaining_demand) & remaining_demand.gt(0)
        # (bs, num_agents ,num_depots)
        mask_depot = mask_depot.unsqueeze(-1).repeat(1, 1, state.num_depots)

            
        # for finished (i.e. no demand and back at depot) instances, mask all but current node
        agent_finished = state.agent_at_depot() & has_no_capacity
        # if only some agents finished, mask all actions for these agents
        mask_depot[agent_finished] = True
        # but it all agents finished, make sure to keep each agent at his current position
        all_agents_finished = agent_finished.all(1) & no_more_demand
        mask_depot[all_agents_finished] = ~(state.current_loc_ohe[...,:state.num_depots][all_agents_finished].bool())
        # (bs, num_agents, num_nodes)
        agent_node_mask = torch.cat((mask_depot, mask_loc_per_agent), 2).bool()

        return agent_node_mask