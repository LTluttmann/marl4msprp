import torch
import torch.nn as nn
from torch.nn.modules import TransformerEncoderLayer
from marlprp.utils.ops import gather_by_index
from marlprp.utils.config import PolicyParams
from marlprp.models.policy_args import TransformerParams, marlprpParams
from marlprp.models.encoder.base import MatNetEncoderOutput


def get_context_emb(params: PolicyParams, extra_key: str = None):

    emb_registry = {
        "marlprp": marlprpContext
    }

    EmbCls = emb_registry.get(params.policy, SimpleContext)

    if extra_key is not None:
        EmbCls = EmbCls[extra_key]

    return EmbCls(params)


class SimpleContext(nn.Module):


    def __init__(self, *args, **kwargs):
        super(SimpleContext, self).__init__()

    def forward(self, td, ma_emb):
        return ma_emb
    

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
    

class marlprpContext(nn.Module):


    def __init__(self, params: marlprpParams):
        super(marlprpContext, self).__init__()
        self.stepwise = params.stepwise_encoding
        if not self.stepwise:
            self.proj_ma_time = nn.Linear(1, params.embed_dim, bias=False)
        if params.use_communication:
            self.communcation_layer = CommunicationLayer(params)
        
    def forward(self, td, ma_emb: torch.Tensor):
        if not self.stepwise:
            # b m 1
            ma_time = td["busy_until"].unsqueeze(-1) / 1000
            # b m d
            ma_time_proj =  self.proj_ma_time(ma_time)
        else:
            ma_time_proj = 0

        ma_emb = ma_emb + ma_time_proj

        if hasattr(self, "communcation_layer"): 
            ma_emb = self.communcation_layer(ma_emb)

        return ma_emb
