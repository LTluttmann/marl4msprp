import torch
import torch.nn as nn
from torch.nn.modules import TransformerEncoderLayer

from marlprp.env.instance import MSPRPState
from marlprp.utils.ops import gather_by_index
from marlprp.utils.config import PolicyParams
from marlprp.models.nn.misc import PositionalEncoding, AttentionGraphPooling
from marlprp.models.policy_args import TransformerParams, MahamParams, HAMParams
from marlprp.models.encoder.utils import MatNetEncoderOutput, ETEncoderOutput


def get_context_emb(params: PolicyParams, key: str = None):

    emb_registry = {
        "maham": {
            "shelf": MahamAgentContext,
            "sku": MahamAgentContext
        },
        "ham": {
            "shelf": SingleAgentContext,
            "sku": SingleAgentContext
        },
        "et": {
            "shelf": EquityTransformerContext,
            "sku": EquityTransformerContext
        },
        "2dptr": {
            "shelf": MultiAgentContext,
            "sku": MultiAgentContext
        },
        "parco": {
            "shelf": MultiAgentContext,
            "sku": MultiAgentContext
        },
    }

    EmbCls = emb_registry.get(params.policy)

    if key is not None:
        EmbCls = EmbCls[key]

    return EmbCls(params)



class MultiAgentContext(nn.Module):

    def __init__(self, params: MahamParams):
        super().__init__()
        self.proj_agent_state = nn.Linear(2, params.embed_dim, bias=params.bias)
        self.proj_agent = nn.Linear(2 * params.embed_dim, params.embed_dim, bias=params.bias)
        if params.use_communication and (params.env.num_agents is None or params.env.num_agents > 1):
            self.comm_layer = CommunicationLayer(params)

    def agent_state_emb(self, state: MSPRPState):
        feats = torch.stack([
            state.remaining_capacity / state.capacity,
            state.tour_length - state.tour_length.max(1, keepdim=True).values
        ], dim=-1)
        state_emb = self.proj_agent_state(feats)

        return state_emb

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState):
        shelf_emb = emb["shelf"]
        current_locs = state.current_location
        # get embedding of current location of agents
        current_loc_emb = gather_by_index(shelf_emb, current_locs, dim=1, squeeze=False)
        # get embedding for agent state
        state_emb = self.agent_state_emb(state)
        # cat and project
        agent_emb = torch.cat((current_loc_emb, state_emb), dim=-1)
        agent_emb = self.proj_agent(agent_emb)
        if hasattr(self, "comm_layer"):
            agent_emb = self.comm_layer(agent_emb, state)
        return agent_emb


class MahamAgentContext(MultiAgentContext):
    def __init__(self, params):
        super().__init__(params)
        self.proj_agent = nn.Linear(3 * params.embed_dim, params.embed_dim, bias=params.bias)
        if params.use_ranking_pe:
            self.pe = PositionalEncoding(embed_dim=params.embed_dim, dropout=params.dropout, max_len=1000)

        # self.problem_proj = nn.Linear(1, params.embed_dim, bias=params.bias)
        self.sku_pooling = AttentionGraphPooling(params)
        self.workload_enc = PositionalEncoding(embed_dim=params.embed_dim, dropout=params.dropout, max_len=1_000)


        
    def graph_emb(self, embs: MatNetEncoderOutput, state: MSPRPState) -> torch.Tensor:
        sku_mask = state.demand.eq(0)
        # (bs, 1)
        remaining_skus = torch.sum(~sku_mask, dim=1, keepdim=True)
        # (bs, 1, d)
        graph_emb = self.sku_pooling(embs["sku"], sku_mask)
        # (bs, 1, d)
        graph_emb = self.workload_enc(graph_emb, remaining_skus)
        return graph_emb.expand(-1,state.num_agents, -1)
    
    
    def agent_state_emb(self, state):
        """add positional encoding"""
        state_emb = super().agent_state_emb(state)
        if hasattr(self, "pe"):
            # rank based on tour length
            agent_ranks = torch.argsort(state.tour_length, dim=1, descending=True)
            state_emb = self.pe(state_emb, agent_ranks, mask=state.agent_pad_mask)
        return state_emb

    # def problem_emb(self, emb: MatNetEncoderOutput, state: MSPRPState):
    #     """Add a graph embedding"""
    #     # bs, 1
    #     remaining_demand = state.demand.sum(1, keepdim=True)
    #     # bs, n_agents
    #     demand_agent_view = (remaining_demand / state.capacity).expand_as(state.remaining_capacity)
    #     # bs, n_sku
    #     # weights = state.demand / (remaining_demand + 1e-6)
    #     # # bs, 1, emb
    #     # weighted_avg = torch.sum(weights.unsqueeze(-1) * emb["sku"], dim=1, keepdim=True)
    #     # # bs, num_agents, emb
    #     # weighted_avg = weighted_avg.expand(-1, state.num_agents, -1)
    #     # bs, num_agents, emb
    #     demand_proj = self.problem_proj(demand_agent_view.unsqueeze(-1))
    #     return demand_proj # + weighted_avg

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState):
        shelf_emb = emb["shelf"]
        current_locs = state.current_location
        # get embedding of current location of agents
        current_loc_emb = gather_by_index(shelf_emb, current_locs, dim=1, squeeze=False)
        # get embedding for agent state
        state_emb = self.agent_state_emb(state)
        problem_emb = self.graph_emb(emb, state)
        # cat and project
        agent_emb = torch.cat((current_loc_emb, state_emb, problem_emb), dim=-1)
        agent_emb = self.proj_agent(agent_emb)
        # if hasattr(self, "pe"):
        #     # rank based on capacity
        #     agent_ranks = torch.argsort(state.remaining_capacity, dim=1, descending=True)
        #     agent_emb = self.pe(agent_emb, agent_ranks, mask=state.agent_pad_mask)
        if hasattr(self, "comm_layer"):
            agent_emb = self.comm_layer(agent_emb, state)
        return agent_emb


class SingleAgentContext(nn.Module):

    def __init__(self, params: HAMParams):
        super().__init__()
        self.proj_agent_state = nn.Linear(3, params.embed_dim, bias=params.bias)
        self.proj_agent = nn.Linear(2 * params.embed_dim, params.embed_dim, bias=params.bias)

    def agent_state_emb(self, state: MSPRPState):
        remaining_capacity = state.remaining_capacity.gather(1, state.active_agent)
        feats = torch.stack([
            remaining_capacity / state.capacity,
            state.demand.sum(1, keepdim=True) / state.capacity,
            state.tour_length.gather(1, state.active_agent)
        ], dim=-1)
        state_emb = self.proj_agent_state(feats)
        return state_emb

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState):
        shelf_emb = emb["shelf"]
        # bs, 1
        current_agents_loc = state.current_location.gather(1, state.active_agent)
        # get embedding of current location of agents
        # bs, 1, emb
        current_loc_emb = gather_by_index(shelf_emb, current_agents_loc, dim=1, squeeze=False)
        # get embedding for agent state
        state_emb = self.agent_state_emb(state)
        # cat and project
        agent_emb = torch.cat((current_loc_emb, state_emb), dim=-1)
        agent_emb = self.proj_agent(agent_emb)
        return agent_emb
    


class EquityTransformerContext(nn.Module):

    def __init__(self, params: MahamParams):
        super().__init__()
        self.proj_distance = nn.Linear(2, params.embed_dim, bias=params.bias)
        self.scale_proj = nn.Linear(1, params.embed_dim, bias=False)
        self.graph_proj = nn.Linear(params.embed_dim, params.embed_dim, bias=False)
        self.loc_proj = nn.Linear(params.embed_dim, params.embed_dim, bias=False)
        self.proj_agent = nn.Linear(4 * params.embed_dim, params.embed_dim, bias=params.bias)


    def _agent_context(self, state: MSPRPState, emb: ETEncoderOutput):
        shelf_emb = emb["shelf"]
        # bs, 1
        current_agents_loc = state.current_location.gather(1, state.active_agent)
        # get embedding of current location of agents
        # bs, 1, emb
        current_loc_emb = gather_by_index(shelf_emb, current_agents_loc, dim=1, squeeze=False)
        # bs, 1, emb
        return self.loc_proj(current_loc_emb)

    def _problem_context(self, state: MSPRPState, emb: ETEncoderOutput):
        # bs, N, emb
        shelf_emb = emb["shelf"]
        # bs, M, emb
        sku_emb = emb["sku"]
        # bs, 1, emb
        graph_emb = torch.cat((shelf_emb, sku_emb), dim=1).mean(1, keepdim=True)
        return self.graph_proj(graph_emb)
    

    def _scale_context(self, state: MSPRPState):
        # bs
        remaining_agents = state.num_valid_agents - (state.active_agent.squeeze(1) + 1)
        # bs
        remaining_agent_capacity = remaining_agents * state.capacity
        # bs
        scale = torch.clamp(state.demand.sum(1) / (remaining_agent_capacity + 1e-6), max=10)
        # bs, emb
        scale_emb = self.scale_proj(scale.unsqueeze(-1))
        return scale_emb.unsqueeze(1)

    def _distance_context(self, state: MSPRPState):
        # bs, 1
        current_tour_len = state.tour_length.gather(1, state.active_agent)
        # bs, 1
        remaining_capacity = state.remaining_capacity.gather(1, state.active_agent)
        # bs, 1, 2
        feats = torch.stack([
            remaining_capacity / state.capacity,
            current_tour_len, 
        ], dim=-1)
        # bs, 1, emb
        state_emb = self.proj_distance(feats)
        return state_emb


    def forward(self, emb: ETEncoderOutput, state: MSPRPState):
        agent_context_emb = self._agent_context(state, emb)
        scale_emb = self._scale_context(state)
        distance_emb = self._distance_context(state)
        problem_emb = self._problem_context(state, emb)
        # cat and project
        # bs, 1, 4*emb
        agent_emb = torch.cat((
            agent_context_emb, 
            scale_emb,
            distance_emb,
            problem_emb
        ), dim=-1)
        agent_emb = self.proj_agent(agent_emb)
        return agent_emb
    


class CommunicationLayer(nn.Module):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.num_heads = params.num_heads
        self.comm_layer = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=params.norm_first,
            batch_first=True,
            bias=params.bias
        )

    def forward(self, x, state: MSPRPState):
        #  bs, num_agents
        # pad_mask = state.agent_pad_mask
        # bs, num_agents, num_agents
        attn_mask = state.remaining_capacity.eq(0).unsqueeze(1).repeat(1, state.num_agents, 1)
        attn_mask = attn_mask.diagonal_scatter(
            torch.full_like(state.current_location, fill_value=False),
            dim1=1, dim2=2
        )
        # add head dimension -> [bs * num_heads, num_agents, num_agents]
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        h = self.comm_layer(x, src_mask=attn_mask) # , src_key_padding_mask=pad_mask)
        return h
