import torch
import torch.nn as nn

from marlprp.env.instance import MSPRPState
from marlprp.models.policy_args import PolicyParams, TransformerParams
from marlprp.models.nn.misc import PositionalEncodingWithOffset
from marlprp.utils.ops import min_max_scale

def get_init_emb_layer(params: PolicyParams):
    init_emb_map = {
        "msprp": MultiAgentInitEmbedding
    }

    emb_cls = init_emb_map[params.env.name]
    init_emb_layer = emb_cls(params)
    return init_emb_layer


class MultiAgentInitEmbedding(nn.Module):
    def __init__(self, policy_params: TransformerParams):
        super(MultiAgentInitEmbedding, self).__init__()
        self.embed_dim = policy_params.embed_dim
        self.capacity = policy_params.env.capacity
        self.depot_proj = nn.Linear(3, policy_params.embed_dim, bias=False)
        self.shelf_proj = nn.Linear(4, policy_params.embed_dim, bias=False)
        self.sku_proj = nn.Linear(2, policy_params.embed_dim, bias=False)

    def _init_depot_embed(self, state: MSPRPState):
        depot_coordinates = state.coordinates[:, :state.num_depots]
        depot_load = state.packing_items
        total_demand = state.demand.sum(1, keepdim=True)
        feats = torch.stack([
            depot_coordinates[..., 0],
            depot_coordinates[..., 1],
            depot_load / total_demand
        ], dim=-1)
        return self.depot_proj(feats)
    

    def _init_shelf_embed(self, state: MSPRPState):
        shelf_coordinates = state.coordinates[:, state.num_depots:]
        num_stored_skus = state.supply.gt(0).sum(-1)
        mean_supply = state.supply.sum(-1) / (num_stored_skus + 1e-6)
        feats = torch.stack([
            shelf_coordinates[..., 0],
            shelf_coordinates[..., 1],
            min_max_scale(num_stored_skus),
            min_max_scale(mean_supply),
        ], dim=-1)
        return self.shelf_proj(feats)

    def _init_sku_embed(self, state: MSPRPState):
        demand_scaled = state.demand.clone() / self.capacity
        num_storage_loc = state.supply.gt(0).sum(1)

        feats = torch.stack([
            demand_scaled,
            min_max_scale(num_storage_loc)
        ], dim=-1)

        return self.sku_proj(feats)
    
    def _init_edge_embed(self, state: MSPRPState):
        supply_scaled = state.supply.clone() / self.capacity
        return supply_scaled
    
    def forward(self, tc):
        depot_emb = self._init_depot_embed(tc)
        shelf_emb = self._init_shelf_embed(tc)
        sku_emb = self._init_sku_embed(tc)
        edge_emb = self._init_edge_embed(tc)
        return depot_emb, shelf_emb, sku_emb, edge_emb
