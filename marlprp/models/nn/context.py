import torch
import torch.nn as nn
from torch.nn.modules import TransformerEncoderLayer

from marlprp.env.instance import MSPRPState
from marlprp.utils.ops import gather_by_index
from marlprp.utils.config import PolicyParams
from marlprp.models.nn.misc import PositionalEncoding
from marlprp.models.policy_args import TransformerParams, MahamParams
from marlprp.models.encoder.base import MatNetEncoderOutput, ETEncoderOutput


def get_context_emb(params: PolicyParams, key: str = None):

    emb_registry = {
        "maham": {
            "shelf": MultiAgentContext,
            "sku": MultiAgentContext
        },
        "ham": {
            "shelf": MultiAgentContext,
            "sku": MultiAgentContext
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


class Context(nn.Module):


    def __init__(self, params: PolicyParams):
        super(Context, self).__init__()

    def forward(self, td):
        pass


class MultiAgentContext(nn.Module):

    def __init__(self, params: MahamParams):
        super().__init__()
        self.proj_agent_state = nn.Linear(3, params.embed_dim, bias=params.bias)
        self.proj_agent = nn.Linear(2 * params.embed_dim, params.embed_dim, bias=params.bias)
        if params.use_communication and (params.env.num_agents is None or params.env.num_agents > 1):
            self.comm_layer = CommunicationLayer(params)

        if params.use_ranking_pe:
            self.pe = PositionalEncoding(embed_dim=params.embed_dim, dropout=params.dropout, max_len=100)


    def agent_state_emb(self, state: MSPRPState):

        feats = torch.stack([
            state.remaining_capacity / state.capacity,
            (state.demand.sum(1, keepdim=True) / state.capacity).expand_as(state.remaining_capacity),
            #state.demand.sum(1, keepdims=True) / (state.remaining_capacity + 1e-7),
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
        if hasattr(self, "pe"):
            agent_ranks = torch.argsort(state.tour_length, dim=1, descending=True)
            agent_emb = self.pe(agent_emb, agent_ranks, mask=state.agent_pad_mask)
        if hasattr(self, "comm_layer"):
            agent_emb = self.comm_layer(agent_emb)
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
        self.comm_layer = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=True,
            batch_first=True,
            bias=params.bias
        )

    def forward(self, x):
        h = self.comm_layer(x)
        return h
