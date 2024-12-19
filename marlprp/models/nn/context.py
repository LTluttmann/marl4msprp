import torch
import torch.nn as nn
from torch.nn.modules import TransformerEncoderLayer
from marlprp.env.instance import MSPRPState
from marlprp.utils.ops import gather_by_index
from marlprp.utils.config import PolicyParams
from marlprp.models.policy_args import TransformerParams, MahamParams
from marlprp.models.encoder.base import MatNetEncoderOutput



def get_context_emb(params: PolicyParams, key: str = None):

    emb_registry = {
        "maham": {
            "shelf": AgentContext,
            "sku": AgentContext
        }
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


class AgentContext(nn.Module):

    def __init__(self, params: MahamParams):
        super().__init__()
        self.proj_agent_state = nn.Linear(1, params.embed_dim, bias=False)
        self.proj_agent = nn.Linear(2 * params.embed_dim, params.embed_dim, bias=False)
        if params.use_communication and (params.env.num_agents is None or params.env.num_agents > 1):
            self.comm_layer = CommunicationLayer(params)

    def agent_state_emb(self, state: MSPRPState):
        feats = torch.stack([
            state.remaining_capacity / state.capacity
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
            agent_emb = self.comm_layer(agent_emb)
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
            norm_first=params.norm_first,
            batch_first=True,
        )

    def forward(self, x):
        h = self.comm_layer(x)
        return h
