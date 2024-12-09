from typing import Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from marlprp.env.generator import MSPRPGenerator
from marlprp.utils.config import EnvParams
from marlprp.env.instance import MSPRPState


class MSPRPEnv:

    name = "msprp"

    def __init__(self, params: EnvParams) -> None:
        self.generator = MSPRPGenerator(params)
        self.params = params


    def reset(self, td: Optional[TensorDict] = None, batch_size=None) -> MSPRPState:
        """Reset function to call at the beginning of each episode"""
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
        if td is None or td.is_empty():
            td = self.generator(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        return self._reset(td)


    def _reset(self, td: TensorDict) -> MSPRPState:
        state = MSPRPState.initialize(
            num_depots=self.params.num_depots,
            **td,
        )
        state.shelf_mask = self.get_node_mask(state)
        return state


    def step(self, actions: torch.Tensor, state: MSPRPState) -> MSPRPState:
        state = state.clone()
        
        actions = actions.split(1, dim=1)

        for action in actions:
            action = action.squeeze(1)
            agent = action["agent"]
            shelf = action.get("shelf", None)
            sku = action.get("sku", None)

            if shelf is not None:
                state = self._update_from_shelf(state, shelf, current_agent=agent)
                state.sku_mask = self.get_sku_mask(state)

            if sku is not None:
                state = self._update_from_sku(state, sku, curr_agent=agent)
                state.shelf_mask = self.get_node_mask(state)

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
        at_depot = next_node.lt(self.params.num_depots)
        depot_id = next_node[at_depot]
        depot_agent = current_agent[at_depot]
        load = self.params.capacity - state.remaining_capacity.gather(1, current_agent[:, None]).squeeze(1)
        state.packing_items[at_depot, depot_id] += load[at_depot]
        # reset remaining load when at depot
        state.remaining_capacity[at_depot, depot_agent] = self.params.capacity
        # after shelf selection the state is in an intermediate phase
        state.is_intermediate = True
        return state

    def _update_from_sku(self, state: MSPRPState, chosen_sku: torch.Tensor, curr_agent: torch.Tensor):
        """
        :param chosen_sku: [BS]
        """
        # Update the dynamic elements differently for if we visit depot vs. a city
        # NOTE: Need to use the curr_node property instead of the shelves determined by the actor, since
        # during beam search different shelves might occur as beam parents for the sku child nodes
        bs = chosen_sku.size(0)
        batch_idx = torch.arange(0, bs, dtype=torch.long, device=state.device)
        chosen_shelf = state.current_location.gather(1, curr_agent[:, None]).squeeze(1)

        depot = state.agent_at_depot(curr_agent)
        visit = ~depot
        # subtract 1 for dummy item
        chosen_sku = chosen_sku[visit] - 1
        agent = curr_agent[visit]
        remaining_capacity = state.remaining_capacity[visit, agent]
        #state = state.masked_select(visit)

        # [BS, 1]
        shelf_idx = chosen_shelf[visit] - state.num_depots

        # [num_visited]
        demand_of_sku = state.demand[visit, chosen_sku]
        supply_at_shelf = state.supply[visit, shelf_idx, chosen_sku]

        taken_units = torch.min(demand_of_sku, supply_at_shelf)
        taken_units = torch.min(taken_units, remaining_capacity)

        state.supply[visit, shelf_idx, chosen_sku] -= taken_units
        state.remaining_capacity[visit, agent] -= taken_units   
        state.demand[visit, chosen_sku] -= taken_units
        # full transition completed from sku selection
        state.is_intermediate = False
        # reset remaining load when at depot
        # state.remaining_capacity[depot, curr_agent[depot]] = self.params.capacity

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
        mask_loc = (state.demand[:, None, :].lt(1e-5) |  state.supply.lt(1e-5)).all(-1)
        mask_loc_per_agent = has_no_capacity[..., None] | mask_loc[:, None]
        # We should avoid traveling to the depot back-to-back, except instance is done
        # (bs, num_agents)
        no_more_demand = state.demand.eq(0).all(1)
        mask_depot = state.agent_at_depot() & ~no_more_demand[:, None]
        # (bs, num_agents, num_depots)
        mask_depot = mask_depot[..., None].expand(-1, state.num_agents, state.num_depots)
        # (bs, num_agents, num_nodes)
        agent_node_mask = torch.cat((mask_depot, mask_loc_per_agent), 2).bool()
        return agent_node_mask


    def get_sku_mask(self, state: MSPRPState, chosen_nodes: TensorDict = None):
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
        nodes = state.current_location.split(1, dim=1)
        # agents = chosen_nodes["agent"]
        # assert torch.allclose(torch.arange(agents.size(1)).view(1, -1).expand_as(agents), agents)
        sku_masks = []
        for node in nodes:

            node = node.squeeze(1)
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

            sku_masks.append(mask)

        sku_masks = torch.stack(sku_masks, dim=1)
        return sku_masks
    

    def get_reward(self, tc: MSPRPState):
        max_travel_distance_per_agent = tc.tour_length.max(1).values

        # calc entropy over packing station utilization
        logp = F.log_softmax(tc.packing_items, dim=-1)
        entropy = -(logp.exp() * logp).sum(1)

        reward = -max_travel_distance_per_agent + self.params.packing_ratio_penalty * entropy
        return reward
