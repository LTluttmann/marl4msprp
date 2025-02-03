import torch
import torch.nn as nn

from marlprp.utils.ops import min_max_scale
from marlprp.env.instance import MSPRPState
from marlprp.models.nn.misc import PositionalEncoding
from marlprp.models.policy_args import PolicyParams, TransformerParams


def get_init_emb_layer(params: PolicyParams):
    init_emb_map = {
        "ham": MultiAgentInitEmbedding,
        "maham": MultiAgentInitEmbedding,
        "2dptr": MultiAgentInitEmbedding,
        "parco": MultiAgentInitEmbedding,
        "et": EquityTransformerInitEmbedding,
    }

    emb_cls = init_emb_map[params.policy]
    init_emb_layer = emb_cls(params)
    return init_emb_layer


class MultiAgentInitEmbedding(nn.Module):
    def __init__(self, policy_params: TransformerParams):
        super(MultiAgentInitEmbedding, self).__init__()
        self.embed_dim = policy_params.embed_dim
        self.scale_supply_by_demand = policy_params.scale_supply_by_demand
        self.depot_proj = nn.Linear(3, policy_params.embed_dim, bias=False)
        self.shelf_proj = nn.Linear(3, policy_params.embed_dim, bias=False)
        self.sku_proj = nn.Linear(2, policy_params.embed_dim, bias=False)

    def _init_depot_embed(self, state: MSPRPState):
        depot_coordinates = state.coordinates[:, :state.num_depots]
        depot_load = state.packing_items
        feats = torch.stack([
            depot_coordinates[..., 0],
            depot_coordinates[..., 1],
            min_max_scale(depot_load, dim=1)
        ], dim=-1)
        return self.depot_proj(feats)
    

    def _init_shelf_embed(self, state: MSPRPState):
        num_skus = state.num_skus
        shelf_coordinates = state.coordinates[:, state.num_depots:]
        num_stored_skus = state.supply.gt(0).sum(-1)
        mean_supply = state.supply.sum(-1) / (num_stored_skus + 1e-6)
        feats = torch.stack([
            shelf_coordinates[..., 0],
            shelf_coordinates[..., 1],
            num_stored_skus / num_skus,
            # mean_supply / state.capacity,
        ], dim=-1)
        return self.shelf_proj(feats)

    def _init_sku_embed(self, state: MSPRPState):
        demand_scaled = state.demand.clone() / state.capacity
        num_storage_loc = state.supply.gt(0).sum(1)

        feats = torch.stack([
            demand_scaled,
            num_storage_loc / state.num_shelves
        ], dim=-1)

        return self.sku_proj(feats)
    
    def _init_edge_embed(self, state: MSPRPState):
        if self.scale_supply_by_demand:
            supply_scaled = torch.where(
                state.demand[:, None].expand_as(state.supply_w_depot) == 0,
                torch.zeros_like(state.supply_w_depot),  # when demand is zero, mask supply
                torch.clamp(state.supply_w_depot.clone() / state.demand[:, None], max=1) # supply more than 100% of demand is irrelevant
            )
        else:
            supply_scaled = state.supply_w_depot.clone() / state.capacity[..., None]
        return supply_scaled
    
    def forward(self, tc: MSPRPState):
        depot_emb = self._init_depot_embed(tc)
        shelf_emb = self._init_shelf_embed(tc)
        sku_emb = self._init_sku_embed(tc)
        edge_emb = self._init_edge_embed(tc)

        node_emb = torch.cat((depot_emb, shelf_emb), dim=1)
        return node_emb, sku_emb, edge_emb


class EquityTransformerInitEmbedding(MultiAgentInitEmbedding):
    def __init__(self, policy_params: TransformerParams):
        super(EquityTransformerInitEmbedding, self).__init__(policy_params)
        self.pe = PositionalEncoding(embed_dim=policy_params.embed_dim)
        self.agent_proj = nn.Linear(1, policy_params.embed_dim, bias=False)

    def _init_agent_embed(self, tc: MSPRPState, depot_emb): 
        bs, num_agents = tc.remaining_capacity.shape
        agent_emb = depot_emb.clone().expand(bs, num_agents, -1)
        agent_order = torch.arange(0, num_agents, device=tc.device).view(1, num_agents).expand(bs, num_agents)
        agent_emb = self.pe(agent_emb, agent_order)
        return agent_emb

    def forward(self, tc: MSPRPState):
        assert tc.num_depots == 1, "ET only implemented for single depot instances yet"
        depot_emb = self._init_depot_embed(tc)
        shelf_emb = self._init_shelf_embed(tc)
        sku_emb = self._init_sku_embed(tc)
        edge_emb = self._init_edge_embed(tc)
        agent_emb = self._init_agent_embed(tc, depot_emb)
        node_emb = torch.cat((depot_emb, shelf_emb), dim=1)
        return node_emb, sku_emb, edge_emb, agent_emb